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
pub struct Fiedler;

impl BranchingDecision for Fiedler {
    fn branch_on(&mut self, g: &Graph, state: &StateManager, component_extractor: &ComponentExtractor, component: ComponentIndex) -> Option<DistributionIndex> {
        let mut best_clause: Option<ClauseIndex> = None;
        let mut best_score = f64::MAX;
        for clause in component_extractor.component_iter(component) {
            if g.clause_has_probabilistic(clause, state) {
                let score = component_extractor.fiedler_value(clause).abs();
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
            if g.clause_has_probabilistic(clause, state) {                
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
