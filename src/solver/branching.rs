//Schlandals
//Copyright (C) 2022 A. Dubray
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

use search_trail::StateManager;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::{ClauseIndex, DistributionIndex, Graph};
use nalgebra::DMatrix;

/// Trait that defined the methods that a branching decision structure must implement.
pub trait BranchingDecision {
    /// Chooses one distribution from
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex>;
    
    fn init(&mut self, g: &Graph, state: &StateManager);
}

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
        let mut laplacian = DMatrix::from_element(cur_idx, cur_idx, 0.0);

        for clause in g.clause_iter() {
            if g.is_clause_constrained(clause, state) {
                for parent in g.parents_clause_iter(clause, state) {
                    if g.is_clause_constrained(parent, state) {
                        laplacian[(lp_idx[clause.0], lp_idx[clause.0])] += 0.5;
                        laplacian[(lp_idx[clause.0], lp_idx[parent.0])] = -1.0;
                    }
                }
                for child in g.children_clause_iter(clause, state) {
                    if g.is_clause_constrained(child, state) {
                        laplacian[(lp_idx[clause.0], lp_idx[clause.0])] += 0.5;
                        laplacian[(lp_idx[clause.0], lp_idx[child.0])] = -1.0;
                    }
                }
            }
        }
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

#[derive(Default)]
pub struct Vsids;

impl BranchingDecision for Vsids {
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
                let score = g.clause_number_unassigned(clause, state);
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
                if score >= best_score {
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

/*
#[cfg(test)]
mod test_simple_branching {
    use super::{BranchingDecision, FirstBranching};
    use crate::core::components::ComponentExtractor;
    use crate::core::graph::{DistributionIndex, Graph};
    use crate::core::trail::StateManager;

    #[test]
    fn test_branching_distribution_order() {
        let _g = Graph::default();
        let _state = StateManager::default();
        let component_extractor = ComponentExtractor::new(&_g, &mut _state);
        let mut distributions: Vec<DistributionIndex> =
            (0..5).map(|i| DistributionIndex(i)).collect();
        let mut b = FirstBranching::default();
        for i in 0..5 {
            assert_eq!(
                Some(DistributionIndex(i)),
                b.branch_on(&_g, &_state, &component_extractor, )
            );
            distributions.remove(0);
        }
        assert_eq!(None, b.branch_on(&_g, &_state, &mut distributions));
    }
}
    */
