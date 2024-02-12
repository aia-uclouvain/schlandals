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
use crate::core::graph::{DistributionIndex, Graph};
use crate::branching::BranchingDecision;
use crate::diagrams::semiring::SemiRing;
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use crate::common::*;
use crate::diagrams::NodeIndex;
use crate::diagrams::dac::dac::Dac;
use crate::diagrams::partial_diagram::pdiagram::{PDiagram, Child};
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
    /// Implication graph of the (Horn) clauses in the input
    graph: Graph,
    /// Manages (save/restore) the states (e.g., reversible primitive types)
    state: StateManager,
    /// Extracts the connected components in the graph
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
    compilation_cache: FxHashMap<CacheKey, Option<NodeIndex>>,
    /// Cache used during the search to store bounds associated with sub-problems
    search_cache: FxHashMap<CacheKey, SearchCacheEntry>,
    /// Statistics gathered during the solving
    statistics: Statistics<S>,
    /// Time limit accorded to the solver
    timeout: u64,
    /// Start time of the solver
    start: Instant,
}

impl<B: BranchingDecision, const S: bool> Solver<B, S> {

    pub fn new(
        graph: Graph,
        state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: Box<B>,
        propagator: Propagator,
        mlimit: u64,
        epsilon: f64,
        timeout: u64,
    ) -> Self {
        Self {
            graph,
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

    /// Solves the problem represented by this solver using a DPLL-search based method.
    pub fn search(&mut self) -> ProblemSolution {
        self.start = Instant::now();
        self.propagator.init(self.graph.number_clauses());
        // First, let's preprocess the problem
        let mut preprocessor = Preprocessor::new(&mut self.graph, &mut self.state, &mut *self.branching_heuristic, &mut self.propagator, &mut self.component_extractor);
        let preproc = preprocessor.preprocess(false);
        if preproc.is_none() {
            return ProblemSolution::Err(Error::Unsat);
        }
        let preproc_in = preproc.unwrap();
        let preproc_out = 1.0 - self.graph.distributions_iter().map(|d| {
            self.graph[d].remaining(&self.state)
        }).product::<f64>();

        // Init the various structures
        self.branching_heuristic.init(&self.graph, &self.state);
        self.propagator.set_forced();
        match self.solve_components(ComponentIndex(0),1, (1.0 + self.epsilon).powf(2.0)) {
            Some((solution, _)) => {
                let (p_in, p_out) = solution.bounds();
                let lb = p_in * preproc_in.clone();
                let ub: Float = 1.0 - (preproc_out + p_out * preproc_in);
                println!("lb {} ub {}", lb, ub);
                let proba = (lb*ub).sqrt();
                self.statistics.print();
                ProblemSolution::Ok(proba)
            },
            None => ProblemSolution::Err(Error::Timeout),
        }
    }

    /// Split the component into multiple sub-components and solve each of them
    fn solve_components(&mut self, component: ComponentIndex, level: isize, bound_factor: f64) -> Option<SearchResult> {
        if PEAK_ALLOC.current_usage_as_mb() as u64 >= self.mlimit {
            self.search_cache.clear();
        }
        self.state.save_state();
        let mut p_in = f128!(1.0);
        let mut p_out = f128!(1.0);
        let mut maximum_probability = f128!(1.0);
        for distribution in self.component_extractor.component_distribution_iter(component) {
            if self.graph[distribution].is_constrained(&self.state) {
                maximum_probability *= self.graph[distribution].remaining(&self.state);
            }
        }

        // If there are no more component to explore (i.e. the sub-problem only contains
        // deterministic variables), then detect_components return false.
        if self.component_extractor.detect_components(&mut self.graph, &mut self.state, component, &mut self.propagator) {
            self.statistics.and_node();
            let number_components = self.component_extractor.number_components(&self.state);
            self.statistics.decomposition(number_components);
            let new_bound_factor = bound_factor.powf(1.0 / number_components as f64);
            for sub_component in self.component_extractor.components_iter(&self.state) {
                if self.start.elapsed().as_secs() >= self.timeout {
                    return None;
                }
                let sub_maximum_probability = self.component_extractor[sub_component].max_probability();
                assert!(0.0 <= sub_maximum_probability && sub_maximum_probability <= 1.0);
                if let Some((sub_problem, backtrack_level)) = self.get_bounds_from_cache(sub_component, new_bound_factor, level) {
                    if backtrack_level != level {
                        let entry = SearchCacheEntry::new((p_in, maximum_probability));
                        self.restore();
                        return Some((entry, backtrack_level));
                    }
                    // If any of the component is not fully explored, then so is the node
                    let (sub_p_in, sub_p_out) = sub_problem.bounds();
                    p_in *= sub_p_in;
                    p_out *= sub_maximum_probability - sub_p_out.clone();
                } else {
                    return None;
                }
            }
        }
        self.restore();
        let entry = SearchCacheEntry::new((p_in, maximum_probability - p_out));
        Some((entry, level - 1))
    }

    /// Retrieves the bounds of a sub-problem from the cache. If the sub-problem has never been
    /// explored or that the bounds, given the bounding factor, are not good enough, the
    /// sub-problem is solved and the result is inserted in the cache.
    fn get_bounds_from_cache(&mut self, component: ComponentIndex, bound_factor: f64, level: isize) -> Option<SearchResult> {
        self.statistics.cache_access();
        let cache_key = self.component_extractor[component].get_cache_key();
        match self.search_cache.get(&cache_key) {
            None => {
                self.statistics.cache_miss();
                if let Some((solution, backtrack_level)) = self.branch(component, level, bound_factor) {
                    self.search_cache.insert(cache_key, solution.clone());
                    Some((solution, backtrack_level))
                } else {
                    None
                }
            },
            Some(cache_entry) => {
                let (p_in, p_out) = cache_entry.bounds();
                if self.are_bounds_tight_enough(p_in, p_out, bound_factor) {
                    Some((cache_entry.clone(), level))
                } else {
                    if let Some((new_solution, backtrack_level)) = self.branch(component, level, bound_factor) {
                        self.search_cache.insert(cache_key, new_solution.clone());
                        Some((new_solution, backtrack_level))
                    } else {
                        None
                    }
                }
            },
        }
    }

    /// Choose a distribution on which to branch, in the sub-problem, and solves the sub-problems
    /// resulting from the branching, recursively.
    /// Returns the bounds of the sub-problem as well as the level to which the solver must
    /// backtrack.
    fn branch(&mut self, component: ComponentIndex, level: isize, bound_factor: f64) -> Option<SearchResult> {
        if let Some(distribution) = self.branching_heuristic.branch_on(&self.graph, &mut self.state, &self.component_extractor, component) {
            self.statistics.or_node();
            let maximum_probability = self.component_extractor[component].max_probability();
            // Stores the accumulated probability of the found models in the sub-problem
            let mut p_in = f128!(0.0);
            // Stores the accumulated probability of the found non-models in the sub-problem
            let mut p_out = f128!(0.0);
            // When a sub-problem is UNSAT, this is the factor that must be used for the
            // computation of p_out
            let unsat_factor = maximum_probability / self.graph[distribution].remaining(&self.state);
            for variable in self.graph[distribution].iter_variables() {
                if self.graph[variable].is_fixed(&self.state) {
                    continue;
                }
                if self.start.elapsed().as_secs() >= self.timeout {
                    return None;
                }
                let v_weight = self.graph[variable].weight().unwrap();
                self.state.save_state();
                match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, component, &mut self.component_extractor, level) {
                    Err(backtrack_level) => {
                        self.statistics.unsat();
                        // The assignment triggered an UNSAT, so the whole sub-problem is part of
                        // the non-models.
                        p_out += v_weight * unsat_factor;
                        if backtrack_level != level {
                            // The clause learning scheme tells us that we need to backtrack
                            // non-chronologically. There are no models in this sub-problem
                            self.restore();
                            return Some((SearchCacheEntry::new((f128!(0.0), f128!(maximum_probability))), backtrack_level));
                        }
                    },
                    Ok(_) => {
                        // No problem during propagation. Before exploring the sub-problems, we can
                        // update the upper bound with the information stored in the propagator
                        // (i.e., the probalistic variables that have been set to false during the
                        // propagation).
                        let p = self.propagator.get_propagation_prob().clone();
                        let removed = unsat_factor - self.component_extractor.component_distribution_iter(component).filter(|d| *d != distribution).map(|d| {
                            self.graph[d].remaining(&self.state)
                        }).product::<f64>();
                        p_out += removed * v_weight;
                        // It is possible that the propagation removes enough variable so that the
                        // bounds are close enough
                        if self.are_bounds_tight_enough(&p_in, &p_out, bound_factor) {
                            self.restore();
                            return Some((SearchCacheEntry::new((p_in, p_out)), level));
                        }
                        if p != 0.0 {
                            if let Some((child_sol, backtrack_level)) = self.solve_components(component, level + 1, bound_factor) {
                                if backtrack_level != level {
                                    self.restore();
                                    return Some((SearchCacheEntry::new((f128!(0.0), f128!(maximum_probability))), backtrack_level));
                                }
                                let (child_p_in, child_p_out) = child_sol.bounds();
                                p_in += child_p_in * &p;
                                p_out += child_p_out * &p;
                                if self.are_bounds_tight_enough(&p_in, &p_out, bound_factor) {
                                    self.restore();
                                    return Some((SearchCacheEntry::new((p_in, p_out)), level));
                                }
                            } else {
                                return None;
                            }
                        }
                    }
                }
                self.restore();
            }
            let cache_entry = SearchCacheEntry::new((p_in, p_out));
            Some((cache_entry, level))
        } else {
            Some((SearchCacheEntry::new((f128!(1.0), f128!(0.0))), level))
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
    pub fn compile<R: SemiRing>(&mut self) -> Option<Dac<R>> {
        // Same as for the search, first we preprocess
        self.propagator.init(self.graph.number_clauses());
        let mut preprocessor = Preprocessor::new(&mut self.graph, &mut self.state, &mut *self.branching_heuristic, &mut self.propagator, &mut self.component_extractor);
        if preprocessor.preprocess(false).is_none() {
            return None;
        }

        self.branching_heuristic.init(&self.graph, &self.state);
        let mut dac = Dac::new();
        match self.expand_prod_node(&mut dac, ComponentIndex(0), 1, (1.0 + self.epsilon).powf(2.0)) {
            None => None,
            Some(_) => {
                dac.optimize_structure();
                Some(dac)
            }
        }
    }

    /// Expands a product node of the arithmetic circuit
    fn expand_prod_node<R: SemiRing>(&mut self, dac: &mut Dac<R>, component: ComponentIndex, level: isize, bound_factor: f64) -> Option<NodeIndex> {
        let mut prod_node = self.get_prod_node_from_propagations(dac);
        if self.component_extractor.detect_components(&mut self.graph, &mut self.state, component, &mut self.propagator) {
            let number_components = self.component_extractor.number_components(&self.state);
            let new_bound_factor = bound_factor.powf(1.0 / number_components as f64);
            for sub_component in self.component_extractor.components_iter(&self.state) {
                if self.start.elapsed().as_secs() >= self.timeout {
                    return None;
                }
                let cache_key = self.component_extractor[component].get_cache_key();
                match self.compilation_cache.get(&cache_key) {
                    Some(node) => {
                        if let Some(n) = node {
                            if prod_node.is_none() {
                                prod_node = Some(dac.add_prod_node());
                            }
                            dac.add_node_output(*n, prod_node.unwrap());
                        } else {
                            return None;
                        }
                    },
                    None => {
                        if let Some(distribution) = self.branching_heuristic.branch_on(&self.graph, &mut self.state, &self.component_extractor, sub_component) {
                        if self.component_extractor[sub_component].has_learned_distribution() {
                                if let Some(child) = self.expand_sum_node(dac, sub_component, distribution, level, new_bound_factor) {
                                    if prod_node.is_none() {
                                        prod_node = Some(dac.add_prod_node());
                                    }
                                    dac.add_node_output(child, prod_node.unwrap());
                                    self.compilation_cache.insert(cache_key, Some(child));
                                } else {
                                    self.compilation_cache.insert(cache_key, None);
                                    return None;
                                }
                            } else {
                                // Still some distributions to branch on, but no more to learn.
                                // This is a partial compilation so we switch to the search solver
                                let maximum_probability = self.component_extractor[sub_component].max_probability();
                                if let Some((child_sol, _)) = self.get_bounds_from_cache(sub_component, new_bound_factor, level) {
                                    let (child_p_in, child_p_out) = child_sol.bounds();
                                    let child_value = (child_p_in * (maximum_probability - child_p_out.clone())).sqrt();
                                    let child = dac.add_approximate_node(child_value.to_f64());
                                    if prod_node.is_none() {
                                        prod_node = Some(dac.add_prod_node());
                                    }
                                    dac.add_node_output(child, prod_node.unwrap());
                                } else {
                                    return None;
                                }
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
        for variable in self.graph[distribution].iter_variables() {
            if self.graph[variable].is_fixed(&self.state) {
                continue;
            }
            if self.start.elapsed().as_secs() >= self.timeout {
                return None;
            }
            self.state.save_state();
            match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, component, &mut self.component_extractor, level) {
                Err(backtrack_level) => {
                    if backtrack_level != level {
                        return None;
                    }
                },
                Ok(_) => {
                    if let Some(child) = self.expand_prod_node(dac, component, level + 1, bounding_factor) {
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
        if self.propagator.has_assignments() || self.propagator.has_unconstrained_distribution() {
            let node = dac.add_prod_node();
            // First, we look at the assignments
            for literal in self.propagator.assignments_iter(&self.state) {
                let variable = literal.to_variable();
                // Only take probabilistic variables set to true
                if self.graph[variable].is_probabilitic() && literal.is_positive() {
                    let distribution = self.graph[variable].distribution().unwrap();
                    // This represent which "probability index" is send to the node
                    let value_index = variable.0 - self.graph[distribution].start().0;
                    let distribution_node = dac.distribution_value_node_index(distribution, value_index, self.graph[variable].weight().unwrap());
                    dac.add_node_output(distribution_node, node);
                }
            }

            // Then, for each unconstrained distribution, we create a sum_node, but only if the
            // distribution has at least one value set to false.
            // Otherwise it would always send 1.0 to the product node.
            for distribution in self.propagator.unconstrained_distributions_iter() {
                if self.graph[distribution].number_false(&self.state) != 0 {
                    let sum_node = dac.add_sum_node();
                    for variable in self.graph[distribution].iter_variables() {
                        if !self.graph[variable].is_fixed(&self.state) {
                            let value_index = variable.0 - self.graph[distribution].start().0;
                            let distribution_node = dac.distribution_value_node_index(distribution, value_index, self.graph[variable].weight().unwrap());
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

// --- LDS ---
impl<B: BranchingDecision, const S: bool> Solver<B, S> {

    pub fn lds(&mut self) -> ProblemSolution {
        self.propagator.init(self.graph.number_clauses());
        // First, let's preprocess the problem
        let mut preprocessor = Preprocessor::new(&mut self.graph, &mut self.state, &mut *self.branching_heuristic, &mut self.propagator, &mut self.component_extractor);
        let preproc = preprocessor.preprocess(false);
        if preproc.is_none() {
            return ProblemSolution::Err(Error::Unsat);
        }
        let preproc_in = preproc.unwrap();
        let preproc_out = 1.0 - self.graph.distributions_iter().map(|d| {
            self.graph[d].remaining(&self.state)
        }).product::<f64>();

        // Init the various structures
        self.branching_heuristic.init(&self.graph, &self.state);
        self.propagator.set_forced();
        let target_epsilon = self.epsilon;

        let mut diagram = PDiagram::new();
        match self.create_and_node(&mut diagram, ComponentIndex(0)) {
            Some(root) => {
                let mut discrepancy = 1;
                loop {
                    self.explore_and_node(&mut diagram, root, 1, (1.0 + self.epsilon).powf(2.0), discrepancy);
                    let (p_in, p_out) = diagram[root].bounds();
                    let lb = p_in * preproc_in.clone();
                    let ub: Float = 1.0 - (preproc_out.clone() + p_out * preproc_in.clone());
                    let best_epsilon = (ub.clone() / lb.clone()).to_f64().sqrt() - 1.0;
                    println!("Discrepancy {} lb {} ub {} discrepancy epsilon {} ({} nodes)", discrepancy, lb, ub, best_epsilon, diagram.number_nodes());
                    let best_proba = (lb*ub).sqrt();
                    if best_epsilon <= target_epsilon {
                        return ProblemSolution::Ok(best_proba);
                    }
                    discrepancy += 1;
                }
            },
            None => {
                let lb = preproc_in.clone();
                let ub = f128!(1.0) - preproc_out.clone();
                let best_epsilon = (ub.clone() / lb.clone()).to_f64().sqrt() - 1.0;
                println!("Problem solved at preprocessing, lb {} ub {} epsilon {}", lb, ub, best_epsilon);
                let best_proba = (lb*ub).sqrt();
                return ProblemSolution::Ok(best_proba);
            },
        }
    }

    fn create_and_node(&mut self, diagram: &mut PDiagram, component: ComponentIndex) -> Option<NodeIndex> {
        let mut maximum_probability = f128!(1.0);
        for distribution in self.component_extractor.component_distribution_iter(component) {
            if self.graph[distribution].is_constrained(&self.state) {
                maximum_probability *= self.graph[distribution].remaining(&self.state);
            }
        }
        if self.component_extractor.detect_components(&mut self.graph, &mut self.state, component, &mut self.propagator) {
            let mut children: Vec<NodeIndex> = vec![];
            for sub_component in self.component_extractor.components_iter(&self.state) {
                let bit_repr = self.component_extractor[sub_component].get_cache_key();
                match self.compilation_cache.get(&bit_repr) {
                    None => {
                        match self.branching_heuristic.branch_on(&self.graph, &mut self.state, &self.component_extractor, sub_component) {
                            None => {},
                            Some(d) => {
                                let max_proba = f128!(self.component_extractor[sub_component].max_probability());
                                let or_node = diagram.add_or_node(d, self.graph[d].number_unfixed(&self.state), max_proba);
                                self.compilation_cache.insert(bit_repr, Some(or_node));
                                children.push(or_node);
                            },
                        }
                    },
                    Some(child) => children.push(child.unwrap()),
                };
            }
            let n = diagram.add_and_node(maximum_probability, children);
            Some(n)
        } else {
            None
        }
    }

    fn explore_and_node(&mut self, diagram: &mut PDiagram, node: NodeIndex, level: isize, bound_factor: f64, discrepancy: usize) {
        let mut p_in = f128!(1.0);
        let mut p_out = f128!(1.0);
        let number_components = self.component_extractor.number_components(&self.state);
        let new_bound_factor = bound_factor.powf(1.0 / number_components as f64);
        for sub_component in self.component_extractor.components_iter(&self.state) {
            let sub_maximum_probability = self.component_extractor[sub_component].max_probability();
            let bit_repr = self.component_extractor[sub_component].get_cache_key();
            if let Some(child) = self.compilation_cache.get(&bit_repr) {
                let child = child.unwrap();
                self.explore_or_node(diagram, child, sub_component, level, new_bound_factor, discrepancy);
                if diagram[child].is_unsat() {
                    // This set the node as unsat and set the bounds to their correct values
                    diagram[node].set_unsat();
                    return;
                }
                p_in *= &diagram[child].bounds().0;
                p_out *= sub_maximum_probability - diagram[child].bounds().1.clone();
            }
        }
        p_out = diagram[node].maximum_probability() - p_out;
        diagram[node].set_bounds((p_in, p_out));
    }

    fn explore_or_node(&mut self, diagram: &mut PDiagram, node: NodeIndex, component: ComponentIndex, level: isize, bound_factor: f64, discrepancy: usize) {
        if diagram[node].is_sat() || diagram[node].is_unsat() {
            return;
        }
        if diagram[node].decision().is_none() {
            match self.branching_heuristic.branch_on(&self.graph, &mut self.state, &self.component_extractor, component) {
                None => {
                    diagram[node].set_sat();
                    return;
                },
                d => diagram[node].set_decision(d),
            };
        }
        let distribution = diagram[node].decision().unwrap();
        let mut child_id = 0;
        let node_start = diagram[node].child_start();
        let mut p_in = f128!(0.0);
        let mut p_out = f128!(0.0);
        let unsat_factor = diagram[node].maximum_probability().to_f64() / self.graph[distribution].remaining(&self.state);
        for variable in self.graph[distribution].iter_variables() {
            if self.graph[variable].is_fixed(&self.state) {
                continue;
            }
            self.state.save_state();
            let node_index = node_start + child_id;
            let v_weight = self.graph[variable].weight().unwrap();
            match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, component, &mut self.component_extractor, level) {
                Err(backtrack_level) => {
                    if backtrack_level != level {
                        diagram[node].set_unsat();
                        self.restore();
                        return;
                    }
                    p_out += v_weight * unsat_factor;
                },
                Ok(_) => {
                    let p = self.propagator.get_propagation_prob().clone();
                    let removed = unsat_factor - self.component_extractor.component_distribution_iter(component).filter(|d| *d != distribution).map(|d| {
                        self.graph[d].remaining(&self.state)
                    }).product::<f64>();
                    p_out += removed * v_weight;
                    if self.are_bounds_tight_enough(&p_in, &p_out, bound_factor) {
                        self.restore();
                        break;
                    }
                    if p != 0.0 {
                        match diagram.get_child_at(node_index) {
                            Child::AndChild(n, _) => {
                                self.component_extractor.detect_components(&mut self.graph, &mut self.state, component, &mut self.propagator);
                                self.explore_and_node(diagram, n, level, bound_factor, discrepancy - child_id);
                                let child_bounds = diagram[n].bounds();
                                p_in += &p * child_bounds.0.clone();
                                p_out += &p * child_bounds.1.clone();
                            },
                            Child::Unexplored => {
                                if let Some(child) = self.create_and_node(diagram, component) {
                                    self.explore_and_node(diagram, child, level + 1, bound_factor, discrepancy - child_id);
                                    let child_bounds = diagram[child].bounds();
                                    p_in += &p * child_bounds.0.clone();
                                    p_out += &p * child_bounds.1.clone();
                                } else {
                                    p_in += &p;
                                }
                            },
                            Child::OrChild(_) => panic!("The children of a OR child is another OR node"),
                        };
                    }
                },
            };
            child_id += 1;
            self.restore();
            if child_id == discrepancy || self.are_bounds_tight_enough(&p_in, &p_out, bound_factor) {
                break;
            }
        }
        diagram[node].set_bounds((p_in, p_out));
    }
}
