//Schlandals
//Copyright (C) 2022-2023 A. Dubray
//
//This program is free software: you can redistribute it and/or modify
//it under the terms of the GNU Affero General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU Affero General Public License for more details.
//
//You should have received a copy of the GNU Affero General Public License
//along with this program.  If not, see <http://www.gnu.org/licenses/>.

//! This module provides several branching heuristics for our solver.
//! The branching heuristics work a bit differently than in classical search-based solvers.
//! Remember that we are solving a _projected weighted model counting_ problems, in which the
//! probabilistic variables are the decision variables (on which the number of models must be counted).
//! In addition to that, we impose that the probabilistic variables are partitionned in distributions, in
//! which the variables are mutually exclusive.
//! This means that the branching decision is reduced to choosing a distribution and then assigning each of its
//! variable to true.
//! Hence, the heuristics provided here returns a distribution (if any are present in the component) instead of a
//! variable.

use search_trail::StateManager;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::{ClauseIndex, DistributionIndex, Graph};
use nalgebra::DMatrix;

/// Trait that defined the methods that a branching decision structure must implement.
pub trait BranchingDecision {
    /// Chooses one distribution from the component to branch on and returns it. If no distribution is present in
    /// the component, returns None.
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex>;
    
    /// Initialize, if necessary, the data structures used by the branching heuristics
    fn init(&mut self, g: &Graph, state: &StateManager);
}

/// A Fiedler-based branching heuristics. The Fiedler vector of a graph is the eigenvector
/// associated with the second smallest eigenvalue of the Laplacian matrix of a graph.
/// This vector gives a value to each node of the graph depending on its position in it.
/// A node on the boundary of the graph has a large value (positive or negative), and a node in the
/// center will have a value close to 0.
///
/// This heuristics computes the fiedler vector for the implication graph of the clauses. Since computing the
/// fiedler vector is computationnaly heavy, it is done only at the beginning, during the initialization.
/// Then, we the branching decision must be selected, the mean fiedler values of all the clauses in the component
/// is computed, and the clause with the value closest to the mean is selected.
/// This means that a clause in the "center" of a component is selected.
/// Then, it selects the first unfixed distribution from the clause.
#[derive(Default)]
pub struct Fiedler {
    fiedler_values: Vec<f64>,
}

impl BranchingDecision for Fiedler {

    fn init(&mut self, g: &Graph, state: &StateManager) {
        let mut lp_idx: Vec<usize> = (0..g.number_clauses()).collect();
        let mut cur_idx = 0;
        for clause in g.clause_iter() {
            if g.is_clause_constrained(clause, state) {
                lp_idx[clause.0] = cur_idx;
                cur_idx += 1;
            }
        }
        
        // Computation of the laplacian matrix of the implication graph. This is a square matrix L in which we have
        // - L[i,i] = the degree of clause i
        // - L[i, j] = -1 if clause i and j are connected (i != j)
        // 
        // We assume that i and j are linked if we have either i -> j or j -> i
        let mut laplacian = DMatrix::from_element(cur_idx, cur_idx, 0.0);

        for clause in g.clause_iter() {
            // We only consider constrained clauses, to avoid noise from unnecessary clauses.
            if g.is_clause_constrained(clause, state) {
                for parent in g.parents_clause_iter(clause, state) {
                    if g.is_clause_constrained(parent, state) {
                        laplacian[(lp_idx[clause.0], lp_idx[clause.0])] += 0.5;
                        laplacian[(lp_idx[parent.0], lp_idx[parent.0])] += 0.5;
                        laplacian[(lp_idx[clause.0], lp_idx[parent.0])] = -1.0;
                    }
                }
                for child in g.children_clause_iter(clause, state) {
                    if g.is_clause_constrained(child, state) {
                        laplacian[(lp_idx[clause.0], lp_idx[clause.0])] += 0.5;
                        laplacian[(lp_idx[child.0], lp_idx[child.0])] += 0.5;
                        laplacian[(lp_idx[clause.0], lp_idx[child.0])] = -1.0;
                    }
                }
            }
        }

        // Computing the eigenvectors
        let decomp = laplacian.hermitian_part().symmetric_eigen();
        let mut smallest = (f64::INFINITY, f64::INFINITY);
        let mut indexes = (0, 0);
        for i in 0..cur_idx {
            let eigenvalue = decomp.eigenvalues[i];
            if eigenvalue < smallest.0 {
                smallest.1 = smallest.0;
                indexes.1 = indexes.0;
                smallest.0 = eigenvalue;
                indexes.0 = i;
            } else if eigenvalue < smallest.1 {
                smallest.1 = eigenvalue;
                indexes.1 = i;
            }
        }
        self.fiedler_values = (0..g.number_clauses()).map(|i| {
            if g.is_clause_constrained(ClauseIndex(i), state) {
                decomp.eigenvectors.row(lp_idx[i])[indexes.1]
            } else {
                0.0
            }
        }).collect::<Vec<f64>>();
    }

    fn branch_on(&mut self, g: &Graph, state: &StateManager, component_extractor: &ComponentExtractor, component: ComponentIndex) -> Option<DistributionIndex> {
        let mut best_clause: Option<ClauseIndex> = None;
        let mut sum_fiedler = 0.0;
        let mut count = 1.0;
        for clause in component_extractor.component_iter(component) {
            if g.is_clause_constrained(clause, state) {
                sum_fiedler += self.fiedler_values[clause.0];
                count += 1.0;
            }
        }
        if count == 0.0 {
            return None;
        }
        let mean_fiedler = sum_fiedler / count;
        let mut best_score = f64::MAX;
        for clause in component_extractor.component_iter(component) {
            if g.is_clause_constrained(clause, state) && g.clause_has_probabilistic(clause, state) {
                let score = (self.fiedler_values[clause.0] - mean_fiedler).abs();
                if score < best_score {
                    best_score = score;
                    best_clause = Some(clause);
                }
            }
        }
        match best_clause {
            Some(clause) => {
                debug_assert!(g.clause_has_probabilistic(clause, state));
                g.get_clause_active_distribution(clause, state)
            },
            None => None
        }
    }
}

/// This heuristic selects the clause with the minimum in degree. In case of tie, it selects the clause
/// for which the less number of parents have been removed.
/// Then, it selects the first unfixed distribution from the clause.
#[derive(Default)]
pub struct MinInDegree;

impl BranchingDecision for MinInDegree {
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let mut best_clause: Option<ClauseIndex> = None;
        let mut best_score = usize::MAX;
        let mut best_tie = usize::MAX;
        for clause in component_extractor.component_iter(component) {
            if g.is_clause_constrained(clause, state) && g.clause_has_probabilistic(clause, state) {                
                let score = g.get_clause_number_parents(clause, state);
                let tie = g.get_clause_removed_parents(clause, state);
                if score < best_score || (score == best_score && tie < best_tie) {
                    best_score = score;
                    best_tie = tie;
                    best_clause = Some(clause);
                }
            }
        }
        
        match best_clause {
            Some(clause) => {
                g.get_clause_active_distribution(clause, state)
            },
            None => None
        }
    }
    
    fn init(&mut self, _g: &Graph, _state: &StateManager) {}
    
}

/// This heuristics selects the clause with the minimum out degree. In case of tie, it selects the clause
/// for which the less number of parents have been removed.
/// Then, it selects the first unfixed distribution from the clause.
#[derive(Default)]
pub struct MinOutDegree;

impl BranchingDecision for MinOutDegree {
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let mut best_clause: Option<ClauseIndex> = None;
        let mut best_score = usize::MAX;
        let mut best_tie = usize::MAX;
        for clause in component_extractor.component_iter(component) {
            if g.is_clause_constrained(clause, state) && g.clause_has_probabilistic(clause, state) {                
                let score = g.get_clause_number_children(clause, state);
                let tie = g.get_clause_removed_children(clause, state);
                if score < best_score || (score == best_score && tie < best_tie) {
                    best_score = score;
                    best_tie = tie;
                    best_clause = Some(clause);
                }
            }
        }
        
        match best_clause {
            Some(clause) => {
                g.get_clause_active_distribution(clause, state)
            },
            None => None
        }
    }
    
    fn init(&mut self, _g: &Graph, _state: &StateManager) {}
    
}

/// This heuristic selects the clause with the maximum degreee.
/// Then, it selects the first unfixed distribution from the clause.
#[derive(Default)]
pub struct MaxDegree;

impl BranchingDecision for MaxDegree {
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let mut best_clause: Option<ClauseIndex> = None;
        let mut best_score = 0;
        for clause in component_extractor.component_iter(component) {
            if g.is_clause_constrained(clause, state) && g.clause_has_probabilistic(clause, state) {                
                let score = g.get_clause_number_children(clause, state) + g.get_clause_number_parents(clause, state);
                if score > best_score {
                    best_score = score;
                    best_clause = Some(clause);
                }
            }
        }
        
        match best_clause {
            Some(clause) => {
                g.get_clause_active_distribution(clause, state)
            },
            None => None
        }
    }
    
    fn init(&mut self, _g: &Graph, _state: &StateManager) {}
    
}

#[cfg(test)]
mod test_heuristics {
    
    use crate::core::graph::{Graph, VariableIndex, DistributionIndex};
    use crate::core::components::ComponentExtractor;
    use crate::solver::branching::*;
    use search_trail::StateManager;

    // Graph used for the tests:
    //
    //          C0 -> C1 ---> C2
    //                 \       |
    //                  \      v 
    //                   \--> C3 --> C4 --> C5
    fn get_graph(state: &mut StateManager) -> Graph {
        let mut g = Graph::new(state);
        let mut ps: Vec<VariableIndex> = vec![];
        for i in 0..6 {
            g.add_distribution(&vec![1.0], state);
            ps.push(VariableIndex(i))
        }
        let ds = (0..6).map(|_| g.add_variable(false, None, None, state)).collect::<Vec<VariableIndex>>();
        // C0
        g.add_clause(ds[0], vec![ps[0]], state);
        // C1
        g.add_clause(ds[1], vec![ds[0], ps[1]], state);
        // C2
        g.add_clause(ds[2], vec![ds[1], ps[2]], state);
        // C3
        g.add_clause(ds[3], vec![ds[1], ds[2], ps[3]], state);
        // C4
        g.add_clause(ds[4], vec![ds[3], ps[4]], state);
        // C5
        g.add_clause(ds[5], vec![ds[4], ps[5]], state);
        g.set_variable(ds[5], false, state);
        g
    }

    #[test]
    fn test_fiedler() {
        let mut state = StateManager::default();
        let g = get_graph(&mut state);
        let extractor = ComponentExtractor::new(&g, &mut state);
        let mut branching = Fiedler::default();
        branching.init(&g, &state);
        let expected_fiedler = vec![-0.506285, -00.297142, -0.216295, -0.0460978, 0.394186, 0.671633];
        println!("{:?}", branching.fiedler_values);
        for i in 0..expected_fiedler.len() {
            assert!((expected_fiedler[i] - branching.fiedler_values[i]).abs() <= 0.00001);
        }
        let decision = branching.branch_on(&g, &state, &extractor, ComponentIndex(0));
        assert!(decision.is_some());
        assert_eq!(DistributionIndex(3), decision.unwrap());
        
    }

    #[test]
    fn test_min_in_degree() {
        let mut state = StateManager::default();
        let g = get_graph(&mut state);
        let extractor = ComponentExtractor::new(&g, &mut state);
        let mut branching = MinInDegree::default();
        let decision = branching.branch_on(&g, &state, &extractor, ComponentIndex(0));
        assert!(decision.is_some());
        assert_eq!(DistributionIndex(0), decision.unwrap());
    }
    
    #[test]
    fn test_min_out_degree() {
        let mut state = StateManager::default();
        let g = get_graph(&mut state);
        let extractor = ComponentExtractor::new(&g, &mut state);
        let mut branching = MinOutDegree::default();
        let decision = branching.branch_on(&g, &state, &extractor, ComponentIndex(0));
        assert!(decision.is_some());
        assert_eq!(DistributionIndex(5), decision.unwrap());
    }
    
    #[test]
    fn test_max_degree() {
        let mut state = StateManager::default();
        let g = get_graph(&mut state);
        let extractor = ComponentExtractor::new(&g, &mut state);
        let mut branching = MaxDegree::default();
        let decision = branching.branch_on(&g, &state, &extractor, ComponentIndex(0));
        assert!(decision.is_some());
        assert_eq!(DistributionIndex(1), decision.unwrap());
    }
}