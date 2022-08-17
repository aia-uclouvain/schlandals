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
use crate::solver::branching::BranchingDecision;
use crate::solver::propagator::SimplePropagator;
use rustc_hash::FxHashMap;

pub struct Solver<S, C, B>
where
    S: StateManager,
    C: ComponentExtractor,
    B: BranchingDecision,
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
    B: BranchingDecision,
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

    /// Returns the projected weighted model count of the component. If this value is in the cache,
    /// returns it immediately and if not compute it
    fn get_cached_component_or_compute(&mut self, component: ComponentIndex) -> f64 {
        let hash = self.component_extractor.get_component_hash(component);
        let should_compute = !self.cache.contains_key(&hash);
        if should_compute {
            let count = self.solve_component(component);
            self.cache.insert(hash, count);
        }
        *self.cache.get(&hash).unwrap()
    }

    /// Computes the projected weighted model count of the component
    fn solve_component(&mut self, component: ComponentIndex) -> f64 {
        let decision = self.branching_heuristic.branch_on(
            &self
                .component_extractor
                .get_component_distributions(component)
                .iter()
                .copied()
                .collect::<Vec<DistributionIndex>>(),
        );
        if let Some(distribution) = decision {
            // The branch objective starts at minus infinity because we use log-probabilities
            let mut branch_obj = f64::NEG_INFINITY;
            for node in self.graph.distribution_iter(distribution) {
                if self.graph.is_node_bound(node, &self.state) {
                    continue;
                }
                self.state.save_state();
                self.graph.propagate_node(node, true, &mut self.state);
                self.component_extractor
                    .detect_components(&self.graph, &mut self.state, component);
                if self.component_extractor.number_components(&self.state) == 0 {
                    // If there are no component in the graph, then the objective of the branch is
                    // the graph objective.
                    branch_obj = (2_f64.powf(branch_obj)
                        + 2_f64.powf(self.graph.get_objective(&self.state)))
                    .log2();
                } else {
                    // Otherwise visit recursively each sub-component and add their objective
                    // to the branch objective
                    for sub_component in self.component_extractor.components_iter(&self.state) {
                        let sub_component_obj = self.get_cached_component_or_compute(sub_component);
                        branch_obj =
                            (2_f64.powf(branch_obj) + 2_f64.powf(sub_component_obj)).log2();
                    }
                }
                self.state.restore_state();
            }
            branch_obj
        } else {
            // If no more decision to make, then the objective of the graph is the probability of
            // the branch
            self.graph.get_objective(&self.state)
        }
    }

    pub fn solve(&mut self) -> f64 {
        let mut obj: f64 = f64::NEG_INFINITY;
        for component in self.component_extractor.components_iter(&self.state) {
            let o = self.get_cached_component_or_compute(component);
            obj = (2_f64.powf(obj) + 2_f64.powf(o)).log2();
        }
        obj
    }
}
