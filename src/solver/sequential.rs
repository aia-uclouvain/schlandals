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

pub struct Solver<'c, 'b, C, B>
where
    C: ComponentExtractor + ?Sized,
    B: BranchingDecision + ?Sized,
{
    graph: Graph,
    state: StateManager,
    component_extractor: &'c mut C,
    branching_heuristic: &'b mut B,
    cache: FxHashMap<u64, f64>,
}

impl<'c, 'b, C, B> Solver<'c, 'b, C, B>
where
    C: ComponentExtractor + ?Sized,
    B: BranchingDecision + ?Sized,
{
    pub fn new(
        graph: Graph,
        state: StateManager,
        component_extractor: &'c mut C,
        branching_heuristic: &'b mut B,
    ) -> Self {
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
        // Need to rethink the hash strategy -> only the nodes is insufficient, need the edges
        let should_compute = !self.cache.contains_key(&hash);
        if should_compute {
            let count = self.solve_component(component);
            self.cache.insert(hash, count);
            count
        } else {
            *self.cache.get(&hash).unwrap()
        }
    }

    /// Computes the projected weighted model count of the component
    fn solve_component(&mut self, component: ComponentIndex) -> f64 {
        let decision = self.branching_heuristic.branch_on(
            &self.graph,
            &self.state,
            &self
                .component_extractor
                .get_component_distributions(component)
                .iter()
                .copied()
                .collect::<Vec<DistributionIndex>>(),
        );
        if let Some(distribution) = decision {
            let mut branch_objectives: Vec<f64> = vec![];
            let mut max_objective = f64::NEG_INFINITY;
            // The branch objective starts at minus infinity because we use log-probabilities
            for node in self.graph.distribution_iter(distribution) {
                self.state.save_state();
                match self.graph.propagate_node(node, true, &mut self.state) {
                    Err(_) => {}
                    Ok(v) => {
                        self.component_extractor.detect_components(
                            &self.graph,
                            &mut self.state,
                            component,
                        );
                        let mut o = v;
                        for sub_component in self.component_extractor.components_iter(&self.state) {
                            o += self.get_cached_component_or_compute(sub_component);
                        }
                        if o != f64::NEG_INFINITY {
                            branch_objectives.push(o);
                            if o > max_objective {
                                max_objective = o;
                            }
                        }
                    }
                };
                self.state.restore_state();
            }
            max_objective
                + branch_objectives
                    .iter()
                    .map(|o| 2_f64.powf(o - max_objective))
                    .sum::<f64>()
                    .log2()
        } else {
            0.0
        }
    }

    pub fn solve(&mut self, current_proba: f64) -> f64 {
        let mut obj = current_proba;
        for component in self.component_extractor.components_iter(&self.state) {
            let o = self.get_cached_component_or_compute(component);
            obj += o;
        }
        obj
    }
}
