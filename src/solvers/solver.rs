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
use crate::diagrams::semiring::SemiRing;
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use crate::common::*;
use crate::diagrams::NodeIndex;
use crate::diagrams::dac::dac::Dac;
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
    /// Cache used in the compilation to store the nodes associated with each sub-problem
    compilation_cache: FxHashMap<CacheKey, NodeIndex>,
    /// Cache used during the search to store bounds associated with sub-problems
    search_cache: FxHashMap<CacheKey, SearchCacheEntry>,
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
            compilation_cache: FxHashMap::default(),
            search_cache: FxHashMap::default(),
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
}

// --- SEARCH --- //

impl<B: BranchingDecision, const S: bool> Solver<B, S> {

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

    /// Split the component into multiple sub-components and solve each of them
    fn solve_components(&mut self, component: ComponentIndex, level: isize, bound_factor: f64, discrepancy: usize) -> (Bounds, isize) {
        if PEAK_ALLOC.current_usage_as_mb() as u64 >= self.mlimit {
            self.search_cache.clear();
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
                let (sub_p_in, sub_p_out) = sub_problem.bounds();
                if backtrack_level != level || sub_p_in.to_f64() == 0.0 {
                    self.restore();
                    return ((F128!(0.0), maximum_probability), if backtrack_level != level { backtrack_level } else { level - 1 });
                }
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
        match self.search_cache.get(&cache_key) {
            None => {
                self.statistics.cache_miss();
                let (solution, backtrack_level) = self.branch(component, level, bound_factor, discrepancy, None);
                self.search_cache.insert(cache_key, solution.clone());
                (solution, backtrack_level)
            },
            Some(cache_entry) => {
                let (p_in, p_out) = cache_entry.bounds();
                if cache_entry.discrepancy() >= discrepancy || self.are_bounds_tight_enough(p_in, p_out, bound_factor) {
                    (cache_entry.clone(), level)
                } else {
                    let (new_solution, backtrack_level) = self.branch(component, level, bound_factor, discrepancy, cache_entry.distribution());
                    self.search_cache.insert(cache_key, new_solution.clone());
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
            self.branching_heuristic.branch_on(&self.problem, &mut self.state, &self.component_extractor, component)
        };
        if let Some(distribution) = decision {
            self.statistics.or_node();
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
                if self.problem[variable].is_fixed(&self.state) {
                    continue;
                }
                if child_id == discrepancy {
                    break;
                }
                let v_weight = self.problem[variable].weight().unwrap();
                if self.start.elapsed().as_secs() >= self.timeout {
                    // Same as above, no more time so we assume that there are no solution in this
                    // branch.
                    break;
                }
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
                        if self.are_bounds_tight_enough(&p_in, &p_out, bound_factor) {
                            self.restore();
                            return (SearchCacheEntry::new((p_in, p_out), discrepancy, Some(distribution)), level);
                        }
                        if p != 0.0 {
                            let new_discrepancy = discrepancy - child_id;
                            let ((child_p_in, child_p_out), backtrack_level) = self.solve_components(component, level + 1, bound_factor, new_discrepancy);
                            if backtrack_level != level {
                                self.restore();
                                return (SearchCacheEntry::new((F128!(0.0), F128!(maximum_probability)), usize::MAX, Some(distribution)), backtrack_level);
                            }
                            p_in += child_p_in * &p;
                            p_out += child_p_out * &p;
                            if self.are_bounds_tight_enough(&p_in, &p_out, bound_factor) {
                                self.restore();
                                return (SearchCacheEntry::new((p_in, p_out), usize::MAX, Some(distribution)), level);
                            }
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

    /// Returns true iff the bounds are close enough so that the probability of the subproblem can
    /// be approximated while assuring that the root node has a valid error-bounded approximation
    fn are_bounds_tight_enough(&self, p_in: &Float, p_out: &Float, bound_factor: f64) -> bool {
        1.0 - p_out.clone() <= p_in.clone() * bound_factor
    }

}

// --- COMPILER --- //

impl<B: BranchingDecision, const S: bool> Solver<B, S> {

    /// Compiles the problem represented by this solver into a (potentially partial) arithmetic
    /// circuit. Return None if the problem is UNSAT.
    pub fn compile<R: SemiRing>(&mut self) -> Dac<R> {
        self.start = Instant::now();
        let mut dac = Dac::new();
        // Same as for the search, first we preprocess
        self.propagator.init(self.problem.number_clauses());
        self.state.save_state();
        let mut preprocessor = Preprocessor::new(&mut self.problem, &mut self.state, &mut self.propagator, &mut self.component_extractor);
        if preprocessor.preprocess().is_none() {
            let root = dac.add_sum_node();
            dac.set_root(root);
            dac.set_compile_time(self.start.elapsed().as_secs());
            return dac;
        }
        let prod_node = self.get_prod_node_from_propagations(&mut dac);
        self.problem.clear_after_preprocess(&mut self.state);
        self.state.restore_state();
        let max_probability = self.problem.distributions_iter().map(|d| self.problem[d].remaining(&self.state)).product::<f64>();
        self.component_extractor.shrink(self.problem.number_clauses(), self.problem.number_variables(), self.problem.number_distributions(), max_probability);
        self.propagator.reduce(self.problem.number_clauses(), self.problem.number_variables());

        self.branching_heuristic.init(&self.problem, &self.state);
        match self.expand_prod_node(&mut dac, ComponentIndex(0), 1, (1.0 + self.epsilon).powf(2.0), prod_node) {
            None => {
                let root = dac.add_sum_node();
                dac.set_root(root);
                dac.set_compile_time(self.start.elapsed().as_secs());
            },
            Some(_) => {
                dac.optimize_structure();
                dac.set_compile_time(self.start.elapsed().as_secs());
            }
        };
        dac
    }

    /// Expands a product node of the arithmetic circuit
    fn expand_prod_node<R: SemiRing>(&mut self, dac: &mut Dac<R>, component: ComponentIndex, level: isize, bound_factor: f64, base_node: Option<NodeIndex>) -> Option<NodeIndex> {
        let mut prod_node = if base_node.is_some() { base_node } else { self.get_prod_node_from_propagations(dac) };
        if self.component_extractor.detect_components(&mut self.problem, &mut self.state, component, &mut self.propagator) {
            let number_components = self.component_extractor.number_components(&self.state);
            let new_bound_factor = bound_factor.powf(1.0 / number_components as f64);
            for sub_component in self.component_extractor.components_iter(&self.state) {
                if self.start.elapsed().as_secs() >= self.timeout {
                    return None;
                }
                let cache_key = self.component_extractor[sub_component].get_cache_key();
                match self.compilation_cache.get(&cache_key) {
                    Some(node) => {
                        if node.0 != usize::MAX {
                            if prod_node.is_none() {
                                prod_node = Some(dac.add_prod_node());
                            }
                            dac.add_node_output(*node, prod_node.unwrap());
                        }
                    },
                    None => {
                        if let Some(distribution) = self.branching_heuristic.branch_on(&self.problem, &mut self.state, &self.component_extractor, sub_component) {
                        if self.component_extractor[sub_component].has_learned_distribution() {
                                if let Some(child) = self.expand_sum_node(dac, sub_component, distribution, level, new_bound_factor) {
                                    if prod_node.is_none() {
                                        prod_node = Some(dac.add_prod_node());
                                    }
                                    dac.add_node_output(child, prod_node.unwrap());
                                    self.compilation_cache.insert(cache_key, child);
                                } else {
                                    self.compilation_cache.insert(cache_key, NodeIndex(usize::MAX));
                                }
                            } else {
                                // Still some distributions to branch on, but no more to learn.
                                // This is a partial compilation so we switch to the search solver
                                let maximum_probability = self.component_extractor[sub_component].max_probability();
                                // TODO: Check when the solver did not solve exactly the
                                // sub-problem ?
                                let (child_sol, _) = self.get_bounds_from_cache(sub_component, new_bound_factor, level, usize::MAX);
                                let (child_p_in, child_p_out) = child_sol.bounds();
                                let child_value = (child_p_in * (maximum_probability - child_p_out.clone())).sqrt();
                                let child = dac.add_approximate_node(child_value.to_f64());
                                if prod_node.is_none() {
                                    prod_node = Some(dac.add_prod_node());
                                }
                                dac.add_node_output(child, prod_node.unwrap());
                            }
                        }
                    },
                };
            }
        }
        if level == 1 && prod_node.is_some() {
            dac.set_root(prod_node.unwrap());
        }
        prod_node
    }

    /// Expand a sum node. Explores all the variables of the distribution, propagate and expand the
    /// sub-circuits.
    fn expand_sum_node<R: SemiRing>(&mut self, dac: &mut Dac<R>, component: ComponentIndex, distribution: DistributionIndex, level: isize, bounding_factor: f64) -> Option<NodeIndex> {
        let mut sum_node: Option<NodeIndex> = None;
        for variable in self.problem[distribution].iter_variables() {
            if self.problem[variable].is_fixed(&self.state) {
                continue;
            }
            if self.start.elapsed().as_secs() >= self.timeout {
                return None;
            }
            self.state.save_state();
            match self.propagator.propagate_variable(variable, true, &mut self.problem, &mut self.state, component, &mut self.component_extractor, level) {
                Err(backtrack_level) => {
                    if backtrack_level != level {
                        return None;
                    }
                },
                Ok(_) => {
                    if let Some(child) = self.expand_prod_node(dac, component, level + 1, bounding_factor, None) {
                        if sum_node.is_none() {
                            sum_node = Some(dac.add_sum_node());
                        }
                        dac.add_node_output(child, sum_node.unwrap());
                    }
                }
            }
            self.restore();
        }
        sum_node
    }

    /// Returns a product node representing the propagation that have been done. In practice, the
    /// product node is constructed from two elements
    ///     1) All the variables that have been set to true
    ///     2) The distributions that have become unconstrained (merged into a sum node)
    fn get_prod_node_from_propagations<R: SemiRing>(&self, dac: &mut Dac<R>) -> Option<NodeIndex> {
        if self.propagator.has_assignments(&self.state) || self.propagator.has_unconstrained_distribution() {
            let node = dac.add_prod_node();
            // First, we look at the assignments
            for literal in self.propagator.assignments_iter(&self.state) {
                let variable = literal.to_variable();
                // Only take probabilistic variables set to true
                if self.problem[variable].is_probabilitic() && literal.is_positive() {
                    let distribution = self.problem[variable].distribution().unwrap();
                    // This represent which "probability index" is send to the node
                    let value_index = variable.0 - self.problem[distribution].start().0;
                    let distribution_node = dac.distribution_value_node_index(distribution, value_index, self.problem[variable].weight().unwrap());
                    dac.add_node_output(distribution_node, node);
                }
            }

            // Then, for each unconstrained distribution, we create a sum_node, but only if the
            // distribution has at least one value set to false.
            // Otherwise it would always send 1.0 to the product node.
            for distribution in self.propagator.unconstrained_distributions_iter() {
                if self.problem[distribution].number_false(&self.state) != 0 {
                    let sum_node = dac.add_sum_node();
                    for variable in self.problem[distribution].iter_variables() {
                        if !self.problem[variable].is_fixed(&self.state) {
                            let value_index = variable.0 - self.problem[distribution].start().0;
                            let distribution_node = dac.distribution_value_node_index(distribution, value_index, self.problem[variable].weight().unwrap());
                            dac.add_node_output(distribution_node, sum_node);
                        }
                    }
                    dac.add_node_output(sum_node, node);
                }
            }
            Some(node)
        } else {
            None
        }
    }
}
