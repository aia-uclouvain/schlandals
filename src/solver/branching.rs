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
use crate::core::graph::DistributionIndex;

pub struct BranchingDecision {
    distributions: Vec<DistributionIndex>,
    next: usize,
}

impl Iterator for BranchingDecision {
    type Item = DistributionIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.distributions.len() {
            None
        } else {
            self.next += 1;
            Some(self.distributions[self.next - 1])
        }
    }
}

pub trait BranchingHeuristic {
    fn decision_from_component<C: ComponentExtractor>(
        &self,
        extractor: &C,
        component: ComponentIndex,
    ) -> Option<BranchingDecision>;
}

pub struct LinearBranching {}

impl LinearBranching {
    pub fn new() -> Self {
        Self {}
    }
}

impl BranchingHeuristic for LinearBranching {
    fn decision_from_component<C: ComponentExtractor>(
        &self,
        extractor: &C,
        component: ComponentIndex,
    ) -> Option<BranchingDecision> {
        let distributions = extractor
            .get_component_distributions(component)
            .iter()
            .copied()
            .collect::<Vec<DistributionIndex>>();
        if distributions.len() == 0 {
            None
        } else {
            Some(BranchingDecision {
                distributions,
                next: 0,
            })
        }
    }
}

#[cfg(test)]
mod test_simple_branching {
    use super::{BranchingHeuristic, LinearBranching};
    use crate::core::components::{ComponentIndex, DFSComponentExtractor};
    use crate::core::graph::{DistributionIndex, Graph};
    use crate::core::trail::TrailedStateManager;

    #[test]
    fn test_branching_distribution_order() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        let w1 = vec![0.3, 0.2, 0.5];
        let w2 = vec![0.1, 0.1, 0.1, 0.7];
        let w3 = vec![0.4, 0.6];

        let nd1 = g.add_distribution(&w1, &mut state);
        let nd2 = g.add_distribution(&w2, &mut state);
        let nd3 = g.add_distribution(&w3, &mut state);

        let n = g.add_node(false, None, None, &mut state);
        g.add_clause(n, &nd1, &mut state);
        g.add_clause(n, &nd2, &mut state);
        g.add_clause(n, &nd3, &mut state);

        let c = DFSComponentExtractor::new(&g, &mut state);
        let b = LinearBranching::new();
        let decisions = b
            .decision_from_component(&c, ComponentIndex(0))
            .unwrap()
            .into_iter()
            .collect::<Vec<DistributionIndex>>();
        assert_eq!(3, decisions.len());
        assert_eq!(DistributionIndex(0), decisions[0]);
        assert_eq!(DistributionIndex(1), decisions[1]);
        assert_eq!(DistributionIndex(2), decisions[2]);
    }
}
