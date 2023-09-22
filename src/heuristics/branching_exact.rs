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
use crate::heuristics::BranchingDecision;


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
    use crate::core::components::ComponentIndex;
    use crate::heuristics::branching_exact::*;
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