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
}

#[derive(Default)]
pub struct Fiedler {
    fiedler_values: Vec<f64>,
}

impl Fiedler {
    pub fn new(g: &Graph, state: &StateManager) -> Self {
        let mut laplacian = DMatrix::from_element(g.number_clauses(), g.number_clauses(), 0.0);
        for clause in g.clause_iter() {
            for parent in g.parents_clause_iter(clause, state) {
                laplacian[(clause.0, clause.0)] += 0.5;
                laplacian[(clause.0, parent.0)] = -1.0;
            }
            for child in g.children_clause_iter(clause, state) {
                laplacian[(clause.0, clause.0)] += 0.5;
                laplacian[(clause.0, child.0)] = -1.0;
            }
        }
        let decomp = laplacian.hermitian_part().symmetric_eigen();
        let mut smallest = (f64::INFINITY, f64::INFINITY);
        let mut indexes = (0, 0);
        for i in 0..g.number_clauses() {
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
        let fiedler_values = (0..g.number_clauses()).map(|i| decomp.eigenvectors.row(i)[indexes.1]).collect::<Vec<f64>>();
        Self {
            fiedler_values
        }
    }
}

impl BranchingDecision for Fiedler {
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
