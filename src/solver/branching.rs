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
    /// This function takes some distributions and returns an option with the next distribution to
    /// branch on (or None if `distributions.len() == 0`)
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex>;
}

/// A simple branching algorithm that selects the first distribution
#[derive(Default)]
pub struct FirstBranching;

impl BranchingDecision for FirstBranching {
    fn branch_on(
        &mut self,
        _g: &Graph,
        _state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let distributions = component_extractor.get_component_distributions(component);
        if distributions.is_empty() {
            None
        } else {
            Some(*distributions.iter().next().unwrap())
        }
    }
}

/// A branching heuristic that uses the sum of the active degree of the nodes in a distribution as
/// heuristic to select the next distribution.
#[derive(Default)]
pub struct ActiveDegreeBranching;

impl BranchingDecision for ActiveDegreeBranching {
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let distributions = component_extractor.get_component_distributions(component);
        let mut distribution: Option<DistributionIndex> = None;
        let mut best_score = -1;
        for d in distributions {
            let active_degree: isize = g
                .distribution_iter(*d)
                .filter(|n| g.is_node_bound(*n, state))
                .map(|n| g.node_number_outgoing(n, state) + g.node_number_incoming(n, state))
                .sum();
            if active_degree > best_score {
                best_score = active_degree;
                distribution = Some(*d);
            }
        }
        distribution
    }
}

/// A branching heuristic that select distribution based on the number of nodes they have on the
/// fringe of the component.
#[derive(Default)]
pub struct Fringe;

impl BranchingDecision for Fringe {
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let distributions = component_extractor.get_component_distributions(component);
        let mut distribution: Option<DistributionIndex> = None;
        let mut best_score = usize::MAX;
        for d in distributions {
            let number_node_fringe = g
                .distribution_iter(*d)
                .filter(|n| g.is_node_bound(*n, state) && g.node_number_outgoing(*n, state) == 0)
                .count();
            if number_node_fringe < best_score {
                best_score = number_node_fringe;
                distribution = Some(*d);
            }
        }
        distribution
    }
}

#[derive(Default)]
pub struct Articulation;

impl BranchingDecision for Articulation {
    fn branch_on(
        &mut self,
        _g: &Graph,
        _state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let distributions = component_extractor.get_component_distributions(component);
        let mut distribution: Option<DistributionIndex> = None;
        let mut best_score = 0;
        for d in distributions {
            let nb_articulation = component_extractor.get_distribution_ap_score(*d);
            if nb_articulation > best_score || distribution.is_none() {
                best_score = nb_articulation;
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
