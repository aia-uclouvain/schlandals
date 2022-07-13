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

use crate::core::graph::{Distribution, Graph};

pub trait BranchingHeuristic {
    fn branching_decision(&mut self, graph: &mut Graph) -> Option<Distribution>;
}

pub struct LinearBranching {
    current: usize,
}

impl LinearBranching {
    pub fn new() -> Self {
        Self { current: 0 }
    }
}

impl BranchingHeuristic for LinearBranching {
    fn branching_decision(&mut self, graph: &mut Graph) -> Option<Distribution> {
        if self.current == graph.distributions.len() {
            None
        } else {
            self.current += 1;
            Some(graph.distributions[self.current - 1])
        }
    }
}

#[cfg(test)]
mod test_simple_branching {
    use super::{BranchingHeuristic, LinearBranching};
    use crate::core::graph::Graph;
    use crate::core::trail::TrailedStateManager;

    #[test]
    fn test_branching_distribution_order() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        let w1 = vec![0.3, 0.2, 0.5];
        let w2 = vec![0.1, 0.1, 0.1, 0.7];
        let w3 = vec![0.4, 0.6];

        g.add_distribution(&w1, &mut state);
        g.add_distribution(&w2, &mut state);
        g.add_distribution(&w3, &mut state);

        let mut b = LinearBranching::new();
        let distributions = g.distributions.clone();
        assert_eq!(distributions[0], b.branching_decision(&mut g).unwrap());
        assert_eq!(distributions[1], b.branching_decision(&mut g).unwrap());
        assert_eq!(distributions[2], b.branching_decision(&mut g).unwrap());
        assert_eq!(None, b.branching_decision(&mut g));
    }
}
