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
use crate::core::graph::*;
use crate::core::trail::*;
use crate::solver::branching::BranchingHeuristic;
use crate::solver::propagator::SimplePropagator;
use rustc_hash::FxHashMap;

pub struct Solver<S, C, B>
where
    S: StateManager,
    C: ComponentExtractor,
    B: BranchingHeuristic,
{
    graph: Graph,
    state: S,
    component_extractor: C,
    branching_heuristic: B,
    cache: FxHashMap<u64, f64>,
}

impl<S, C, B> Solver<S, C, B>
where
    S: StateManager,
    C: ComponentExtractor,
    B: BranchingHeuristic,
{
    pub fn new(graph: Graph, state: S, component_extractor: C, branching_heuristic: B) -> Self {
        Self {
            graph,
            state,
            component_extractor,
            branching_heuristic,
            cache: FxHashMap::default(),
        }
    }

    fn get_cached_component_or_compute(&mut self, component: ComponentIndex) -> f64 {
        let hash = self.component_extractor.get_component_hash(component);
        let should_compute = !self.cache.contains_key(&hash);
        if should_compute {
            let count = self.solve_component(component);
            self.cache.insert(hash, count);
        }
        *self.cache.get(&hash).unwrap()
    }

    fn solve_component(&mut self, component: ComponentIndex) -> f64 {
        let decisions = self
            .branching_heuristic
            .decision_from_component(&self.component_extractor, component);
        if let Some(branching) = decisions {
            let mut obj = 0.0;
            for d in branching {
                for node in self.graph.distribution_iter(d) {
                    self.state.save_state();
                    self.graph.propagate_node(node, true, &mut self.state);
                    self.component_extractor.detect_components(
                        &self.graph,
                        &mut self.state,
                        component,
                    );
                    for sub_component in self.component_extractor.components_iter(&self.state) {
                        obj += self.get_cached_component_or_compute(sub_component);
                    }
                    self.state.restore_state();
                }
            }
            obj
        } else {
            self.graph.get_objective(&self.state)
        }
    }

    pub fn solve(&mut self) -> f64 {
        let mut obj = 0.0;
        for component in self.component_extractor.components_iter(&self.state) {
            obj += self.get_cached_component_or_compute(component);
        }
        obj
    }
}
