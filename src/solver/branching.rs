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

use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::{DistributionIndex, Graph};
use crate::core::trail::StateManager;

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
pub struct ChildrenFiedlerAvg;

impl BranchingDecision for ChildrenFiedlerAvg {
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let distributions = component_extractor.get_component_distributions(component);
        let mut distribution: Option<DistributionIndex> = None;
        let mut best_score = f64::INFINITY;
        for d in distributions {
            let mut sum = 0.0;
            let mut count = 0.0;
            for node_value in g.distribution_iter(*d).filter(|n| !g.is_node_bound(*n, state)).map(|n| component_extractor.average_children_fiedler(g, n, state)) {
                if let Some(v) = node_value {
                    sum += v;
                    count += 1.0;
                }
            }
            let score = sum / count;
            if score < best_score {
                best_score = score;
                distribution = Some(*d);
            }
        }
        distribution
    }
}

#[derive(Default)]
pub struct ChildrenFiedlerMin;

impl BranchingDecision for ChildrenFiedlerMin {
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let distributions = component_extractor.get_component_distributions(component);
        let mut distribution: Option<DistributionIndex> = None;
        let mut best_score = f64::INFINITY;
        for d in distributions {
            let mut score = f64::INFINITY;
            for node_value in g.distribution_iter(*d).filter(|n| !g.is_node_bound(*n, state)).map(|n| component_extractor.minimum_children_fiedler(g, n, state)) {
                if node_value < score {
                    score = node_value;
                }
            }
            if score < best_score {
                best_score = score;
                distribution = Some(*d);
            }
        }
        distribution
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
