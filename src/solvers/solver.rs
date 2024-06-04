//Schlandal
//Copyright (C) 2022-2024 A. Dubray, L. Dierckx
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
use rustc_hash::FxHashMap;
use search_trail::{StateManager, SaveAndRestore};

use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::problem::{DistributionIndex, Problem};
use crate::branching::BranchingDecision;
use crate::diagrams::dac::dac::Dac;
use crate::diagrams::semiring::SemiRing;
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use crate::common::*;
use crate::PEAK_ALLOC;
use super::statistics::Statistics;
use rug::Float;
use super::*;
use std::time::Instant;

type SearchResult = (SearchCacheEntry, isize);

/// This structure represent a general solver in Schlandals. It stores a representation of the
/// problem and various structure that are used when solving it.
/// It has two solving strategies:
///     1. A modified DPLL search over the distributions of the problem
///     2. A compiler which run the DPLL search but store the trace as an arithemtic circuit.
/// It is also possible to run the solver in an hybrid mode. That is, the solver starts with a
/// compilation part and then switch to a search for some sub-problems.
///
/// The solver supports epsilon-approximation for the search, providing an approximate probability
/// with bounded error.
/// Given a probability p and an approximate probability p', we say that p' is an epsilon-bounded
/// approximation iff
///     p / (1 + epsilon) <= p' <= p*(1 + epsilon)
///
/// Finally, the compiler is able to create an arithmetic circuit for any semi-ring. Currently
/// implemented are the probability semi-ring (the default) and tensor semi-ring, which uses torch
/// tensors (useful for automatic differentiation in learning).
pub struct Solver<B: BranchingDecision, const S: bool> {
    /// Implication problem of the (Horn) clauses in the input
    problem: Problem,
    /// Manages (save/restore) the states (e.g., reversible primitive types)
    state: StateManager,
    /// Extracts the connected components in the problem
    component_extractor: ComponentExtractor,
    /// Heuristics that decide on which distribution to branch next
    branching_heuristic: Box<B>,
    /// Runs Boolean Unit Propagation and Schlandals' specific propagation at each decision node
    propagator: Propagator,
    /// Error factor when running an approximate search over a sub-problem. If equals to 0.0, then
    /// run an exact search
    epsilon: f64,
    /// Memory limit, in megabytes, when running a search over a sub-problem
    mlimit: u64,
    cache: FxHashMap<CacheKey, SearchCacheEntry>,
    /// Statistics gathered during the solving
    statistics: Statistics<S>,
    /// Time limit accorded to the solver
    timeout: u64,
    /// Start time of the solver
    start: Instant,
    /// Product of the weight of the variables set to true during propagation
    preproc_in: Option<Float>,
    /// Probability of removed interpretation during propagation
    preproc_out: Option<f64>,
}

impl<B: BranchingDecision, const S: bool> Solver<B, S> {

    pub fn new(
        problem: Problem,
        state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: Box<B>,
        propagator: Propagator,
        mlimit: u64,
        epsilon: f64,
        timeout: u64,
    ) -> Self {
        Self {
            problem,
            state,
            component_extractor,
            branching_heuristic,
            propagator,
            epsilon,
            mlimit,
            cache: FxHashMap::default(),
            statistics: Statistics::default(),
            timeout,
            start: Instant::now(),
            preproc_in: None,
            preproc_out: None,
        }
    }

    /// Restores the state of the solver to the previous state
    fn restore(&mut self) {
        self.propagator.restore(&self.state);
        self.state.restore_state();
    }

    /// Solves the problem represented by this solver using a DPLL-search based method.
    pub fn search(&mut self, is_lds: bool) -> Solution {
        if !is_lds {
            let sol = self.do_discrepancy_iteration(usize::MAX);
            self.statistics.print();
            sol
        } else {
            let mut discrepancy = 1;
            loop {
                let solution = self.do_discrepancy_iteration(discrepancy);
                solution.print();
                if self.start.elapsed().as_secs() >= self.timeout || solution.has_converged(self.epsilon) {
                    self.statistics.print();
                    return solution;
                }
                discrepancy += 1;
            }
        }
    }

    pub fn compile(&mut self) -> Dac<Float> {
        Dac::<Float>::new()
    }

    pub fn do_discrepancy_iteration(&mut self, discrepancy: usize) -> Solution {
        if self.preproc_in.is_none() {
            self.start = Instant::now();
            self.propagator.init(self.problem.number_clauses());
            // First, let's preprocess the problem
            self.state.save_state();
            let mut preprocessor = Preprocessor::new(&mut self.problem, &mut self.state, &mut self.propagator, &mut self.component_extractor);
            let preproc = preprocessor.preprocess();
            if preproc.is_none() {
                return Solution::new(F128!(0.0), F128!(0.0), self.start.elapsed().as_secs());
            }
            self.preproc_in = Some(preproc.unwrap());
            self.preproc_out = Some(1.0 - self.problem.distributions_iter().map(|d| {
                self.problem[d].remaining(&self.state)
            }).product::<f64>());
            self.problem.clear_after_preprocess(&mut self.state);
            self.state.restore_state();
            if self.problem.number_clauses() == 0 {
                let lb = self.preproc_in.clone().unwrap();
                let ub = F128!(1.0 - self.preproc_out.unwrap());
                return Solution::new(lb, ub, self.start.elapsed().as_secs());
            }
            let max_probability = self.problem.distributions_iter().map(|d| self.problem[d].remaining(&self.state)).product::<f64>();
            self.component_extractor.shrink(self.problem.number_clauses(), self.problem.number_variables(), self.problem.number_distributions(), max_probability);
            self.propagator.reduce(self.problem.number_clauses(), self.problem.number_variables());

            // Init the various structures
            self.branching_heuristic.init(&self.problem, &self.state);
            self.propagator.set_forced();
        }
        let ((p_in, p_out), _) = self.solve_components(ComponentIndex(0),1, (1.0 + self.epsilon).powf(2.0), discrepancy);
        let lb = p_in * self.preproc_in.clone().unwrap();
        let ub: Float = 1.0 - (self.preproc_out.unwrap() + p_out * self.preproc_in.clone().unwrap());
        Solution::new(lb, ub, self.start.elapsed().as_secs())
    }

    /// Split the component into multiple sub-components and solve each of them
    fn solve_components(&mut self, component: ComponentIndex, level: isize, bound_factor: f64, discrepancy: usize) -> (Bounds, isize) {
        if PEAK_ALLOC.current_usage_as_mb() as u64 >= self.mlimit {
            self.cache.clear();
        }
        self.state.save_state();
        let mut p_in = F128!(1.0);
        let mut p_out = F128!(1.0);
        let mut maximum_probability = F128!(1.0);
        for distribution in self.component_extractor.component_distribution_iter(component) {
            if self.problem[distribution].is_constrained(&self.state) {
                maximum_probability *= self.problem[distribution].remaining(&self.state);
            }
        }

        // If there are no more component to explore (i.e. the sub-problem only contains
        // deterministic variables), then detect_components return false.
        if self.component_extractor.detect_components(&mut self.problem, &mut self.state, component, &mut self.propagator) {
            self.statistics.and_node();
            let number_components = self.component_extractor.number_components(&self.state);
            self.statistics.decomposition(number_components);
            let new_bound_factor = bound_factor.powf(1.0 / number_components as f64);
            for sub_component in self.component_extractor.components_iter(&self.state) {
                // If the solver has no more time, assume that there are no solutions in the
                // remaining of the components. This way we always produce a valid lower/upper
                // bound.
                if self.start.elapsed().as_secs() >= self.timeout {
                    return ((F128!(0.0), F128!(0.0)), level - 1);
                }
                let sub_maximum_probability = self.component_extractor[sub_component].max_probability();
                assert!(0.0 <= sub_maximum_probability && sub_maximum_probability <= 1.0);
                let (sub_problem, backtrack_level) = self.get_bounds_from_cache(sub_component, new_bound_factor, level, discrepancy);
                if backtrack_level != level {
                    self.restore();
                    return ((F128!(0.0), maximum_probability), backtrack_level);
                }
                // If any of the component is not fully explored, then so is the node
                let (sub_p_in, sub_p_out) = sub_problem.bounds();
                p_in *= sub_p_in;
                p_out *= sub_maximum_probability - sub_p_out.clone();
            }
        }
        self.restore();
        ((p_in, maximum_probability - p_out), level - 1)
    }

    /// Retrieves the bounds of a sub-problem from the cache. If the sub-problem has never been
    /// explored or that the bounds, given the bounding factor, are not good enough, the
    /// sub-problem is solved and the result is inserted in the cache.
    fn get_bounds_from_cache(&mut self, component: ComponentIndex, bound_factor: f64, level: isize, discrepancy: usize) -> SearchResult {
        self.statistics.cache_access();
        let cache_key = self.component_extractor[component].get_cache_key();
        match self.cache.get(&cache_key) {
            None => {
                self.statistics.cache_miss();
                let (solution, backtrack_level) = self.branch(component, level, bound_factor, discrepancy, None);
                self.cache.insert(cache_key, solution.clone());
                (solution, backtrack_level)
            },
            Some(cache_entry) => {
                let (p_in, p_out) = cache_entry.bounds();
                let max_proba = self.component_extractor[component].max_probability();
                if cache_entry.discrepancy() >= discrepancy || ((p_in.to_f64() - (max_proba - p_out.to_f64())).abs() <= FLOAT_CMP_THRESHOLD) {
                    (cache_entry.clone(), level)
                } else {
                    let (new_solution, backtrack_level) = self.branch(component, level, bound_factor, discrepancy, cache_entry.distribution());
                    self.cache.insert(cache_key, new_solution.clone());
                    (new_solution, backtrack_level)
                }
            },
        }
    }

    /// Choose a distribution on which to branch, in the sub-problem, and solves the sub-problems
    /// resulting from the branching, recursively.
    /// Returns the bounds of the sub-problem as well as the level to which the solver must
    /// backtrack.
    fn branch(&mut self, component: ComponentIndex, level: isize, bound_factor: f64, discrepancy: usize, choice: Option<DistributionIndex>) -> SearchResult {
        let decision = if choice.is_some() {
            choice
        } else {
            self.statistics.or_node();
            self.branching_heuristic.branch_on(&self.problem, &mut self.state, &self.component_extractor, component)
        };
        if let Some(distribution) = decision {
            let maximum_probability = self.component_extractor[component].max_probability();
            // Stores the accumulated probability of the found models in the sub-problem
            let mut p_in = F128!(0.0);
            // Stores the accumulated probability of the found non-models in the sub-problem
            let mut p_out = F128!(0.0);
            // When a sub-problem is UNSAT, this is the factor that must be used for the
            // computation of p_out
            let unsat_factor = maximum_probability / self.problem[distribution].remaining(&self.state);
            let mut child_id = 0;
            for variable in self.problem[distribution].iter_variables() {
                if self.start.elapsed().as_secs() >= self.timeout {
                    break;
                }
                if self.problem[variable].is_fixed(&self.state) {
                    continue;
                }
                if child_id == discrepancy {
                    break;
                }
                let v_weight = self.problem[variable].weight().unwrap();
                self.state.save_state();
                match self.propagator.propagate_variable(variable, true, &mut self.problem, &mut self.state, component, &mut self.component_extractor, level) {
                    Err(backtrack_level) => {
                        self.statistics.unsat();
                        // The assignment triggered an UNSAT, so the whole sub-problem is part of
                        // the non-models.
                        p_out += v_weight * unsat_factor;
                        if backtrack_level != level {
                            // The clause learning scheme tells us that we need to backtrack
                            // non-chronologically. There are no models in this sub-problem
                            self.restore();
                            return (SearchCacheEntry::new((F128!(0.0), F128!(maximum_probability)), usize::MAX, Some(distribution)), backtrack_level);
                        }
                    },
                    Ok(_) => {
                        // No problem during propagation. Before exploring the sub-problems, we can
                        // update the upper bound with the information stored in the propagator
                        // (i.e., the probalistic variables that have been set to false during the
                        // propagation).
                        let p = self.propagator.get_propagation_prob().clone();
                        let removed = unsat_factor - self.component_extractor.component_distribution_iter(component).filter(|d| *d != distribution).map(|d| {
                            self.problem[d].remaining(&self.state)
                        }).product::<f64>();
                        p_out += removed * v_weight;
                        // It is possible that the propagation removes enough variable so that the
                        // bounds are close enough
                        if p != 0.0 {
                            let new_discrepancy = discrepancy - child_id;
                            let ((child_p_in, child_p_out), backtrack_level) = self.solve_components(component, level + 1, bound_factor, new_discrepancy);
                            if backtrack_level != level {
                                self.restore();
                                return (SearchCacheEntry::new((F128!(0.0), F128!(maximum_probability)), usize::MAX, Some(distribution)), backtrack_level);
                            }
                            p_in += child_p_in * &p;
                            p_out += child_p_out * &p;
                        }
                    }
                }
                self.restore();
                child_id += 1;
            }
            let cache_entry = SearchCacheEntry::new((p_in, p_out), discrepancy, Some(distribution));
            (cache_entry, level)
        } else {
            (SearchCacheEntry::new((F128!(1.0), F128!(0.0)), usize::MAX, None), level)
        }
    }
}
