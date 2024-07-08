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
use search_trail::{SaveAndRestore, StateManager};

use crate::statistics::Statistics;
use crate::branching::BranchingDecision;
use crate::common::*;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::problem::{DistributionIndex, Problem, VariableIndex};
use crate::ac::ac::{NodeIndex, Dac};
use crate::semiring::SemiRing;
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use crate::PEAK_ALLOC;
use rug::Float;
use std::time::Instant;

type SearchResult = (SearchCacheEntry, isize);
type DistributionChoice = (DistributionIndex, VariableIndex);
type UnconstrainedDistribution = (DistributionIndex, Vec<VariableIndex>);

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
        self.start = Instant::now();
        self.state.save_state();
        if let Some(sol) = self.preprocess() {
            return sol;
        }
        self.restructure_after_preprocess();

        self.pwmc(is_lds)
    }

    fn pwmc(&mut self, is_lds: bool) -> Solution {
        if self.problem.number_clauses() == 0 {
            let lb = self.preproc_in.clone().unwrap();
            let ub = F128!(1.0 - self.preproc_out.unwrap());
            return Solution::new(lb, ub, self.start.elapsed().as_secs());
        }
        if !is_lds {
            let sol = self.do_discrepancy_iteration(usize::MAX);
            self.statistics.print();
            sol
        } else {
            let mut discrepancy = 1;
            let mut prev_time = 0;
            loop {
                let solution = self.do_discrepancy_iteration(discrepancy);
                solution.print();
                if self.start.elapsed().as_secs() >= self.timeout || solution.has_converged(self.epsilon) || 2*self.start.elapsed().as_secs() - prev_time >= self.timeout {
                    self.statistics.print();
                    return solution;
                }
                prev_time = self.start.elapsed().as_secs();
                discrepancy += 1;
            }
        }
    }

    /// Preprocess the problem, if the problem is solved during the preprocess, return a solution.
    /// Returns None otherwise
    fn preprocess(&mut self) -> Option<Solution> {
        self.propagator.init(self.problem.number_clauses());
        let mut preprocessor = Preprocessor::new(
            &mut self.problem,
            &mut self.state,
            &mut self.propagator,
            &mut self.component_extractor,
        );
        let preproc = preprocessor.preprocess();
        if preproc.is_none() {
            return Some(Solution::new(
                F128!(0.0),
                F128!(0.0),
                self.start.elapsed().as_secs(),
            ));
        }
        self.preproc_in = Some(preproc.unwrap());
        let max_after_preproc= self.problem.distributions_iter().map(|d| {
            self.problem[d].remaining(&self.state)
        }).product::<f64>();
        self.preproc_out = Some(1.0 - max_after_preproc);
        None
    }

    fn restructure_after_preprocess(&mut self) {
        self.problem.clear_after_preprocess(&mut self.state);
        self.state.restore_state();
        let max_probability = self
            .problem
            .distributions_iter()
            .map(|d| self.problem[d].remaining(&self.state))
            .product::<f64>();
        self.component_extractor.shrink(
            self.problem.number_clauses(),
            self.problem.number_variables(),
            self.problem.number_distributions(),
            max_probability,
        );
        self.propagator.reduce(
            self.problem.number_clauses(),
            self.problem.number_variables(),
        );

        // Init the various structures
        self.branching_heuristic.init(&self.problem, &self.state);
    }

    pub fn do_discrepancy_iteration(&mut self, discrepancy: usize) -> Solution {
        let (result,_) = self.get_bounds_from_cache(ComponentIndex(0), (1.0+self.epsilon).powf(2.0), 1, discrepancy);
        let p_in = result.bounds().0.clone();
        let p_out = result.bounds().1.clone();
        let lb = p_in * self.preproc_in.clone().unwrap();
        let ub: Float =
        1.0 - (self.preproc_out.unwrap() + p_out * self.preproc_in.clone().unwrap());
        Solution::new(lb, ub, self.start.elapsed().as_secs())
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
                    (cache_entry.clone(), level - 1)
                } else {
                    let (new_solution, backtrack_level) = self.branch(
                        component,
                        level,
                        bound_factor,
                        discrepancy,
                        cache_entry.distribution(),
                    );
                    self.cache.insert(cache_key, new_solution.clone());
                    (new_solution, backtrack_level)
                }
            }
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
            let mut children_map: FxHashMap<VariableIndex, CacheChildren> = FxHashMap::default();
            for variable in self.problem[distribution].iter_variables() {
                if self.problem[variable].is_fixed(&self.state) {
                    continue;
                }
                if self.start.elapsed().as_secs() >= self.timeout || child_id == discrepancy {
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
                            return (SearchCacheEntry::new((F128!(0.0), F128!(maximum_probability)), usize::MAX, Some(distribution), FxHashMap::default()), backtrack_level);
                        }
                    },
                    Ok(_) => {
                        // No problem during propagation. Before exploring the sub-problems, we can
                        // update the upper bound with the information stored in the propagator
                        // (i.e., the probalistic variables that have been set to false during the
                        // propagation).
                        let p = self.propagator.get_propagation_prob().clone();
                        let removed = unsat_factor
                        - self
                            .component_extractor
                            .component_distribution_iter(component)
                            .filter(|d| *d != distribution)
                            .map(|d| self.problem[d].remaining(&self.state))
                            .product::<f64>();
                        p_out += removed * v_weight;
                        let new_discrepancy = discrepancy - child_id;
                        // Now that we've branched and applied the propagation, we split the
                        // remaining problem into independent sub-components and solve them
                        // independently.
                        // Their bounds must be multiplied by each other, this is a product node.
                        let mut prod_p_in = F128!(1.0);
                        let mut prod_p_out = F128!(1.0);
                        let mut prod_maximum_probability = F128!(1.0);

                        if PEAK_ALLOC.current_usage_as_mb() as u64 >= self.mlimit {
                            self.cache.clear();
                        }
                        for child_distribution in self.component_extractor.component_distribution_iter(component) {
                            if self.problem[child_distribution].is_constrained(&self.state) {
                                prod_maximum_probability *= self.problem[child_distribution].remaining(&self.state);
                            }
                        }
                        let (forced_distribution_var, unconstrained_distribution_var) = self.forced_from_propagation();
                        let mut child_entry = CacheChildren::new(forced_distribution_var, unconstrained_distribution_var);

                        self.state.save_state();
                        // If there are no more component to explore (i.e. the sub-problem only contains
                        // deterministic variables), then detect_components return false.
                        let mut is_node_sat = true;
                        if self.component_extractor.detect_components(&mut self.problem, &mut self.state, component) {
                            self.statistics.and_node();
                            let number_components = self.component_extractor.number_components(&self.state);
                            self.statistics.decomposition(number_components);
                            let new_bound_factor = bound_factor.powf(1.0 / number_components as f64);
                            for sub_component in self.component_extractor.components_iter(&self.state) {
                                // If the solver has no more time, assume that there are no solutions in the
                                // remaining of the components. This way we always produce a valid lower/upper
                                // bound.
                                if self.start.elapsed().as_secs() >= self.timeout {
                                    return (SearchCacheEntry::new((p_in, p_out), usize::MAX, Some(distribution), children_map), level - 1);
                                }
                                let sub_maximum_probability = self.component_extractor[sub_component].max_probability();
                                let (sub_problem, backtrack_level) = self.get_bounds_from_cache(sub_component, new_bound_factor, level +1, new_discrepancy); // the function was called with level + 1 and level was given

                                if backtrack_level != level { // the function was called with level + 1 and level was used here
                                    // Or node is unsat
                                    self.restore();
                                    return (SearchCacheEntry::new((F128!(0.0),
                                        F128!(maximum_probability)),
                                        usize::MAX,
                                        Some(distribution),
                                        FxHashMap::default()), backtrack_level);
                                }
                                // If any of the component is not fully explored, then so is the node
                                let (sub_p_in, sub_p_out) = sub_problem.bounds();
                                prod_p_in *= sub_p_in;
                                prod_p_out *= sub_maximum_probability - sub_p_out.clone();
                                // In case prod_p_in is 0, it means that one of the component has
                                // probability 0, hence we do not need to explore further.
                                // Notice that we do not need to set prod_p_out to the maximum
                                // probability. After the loop, we reassign prod_p_out using the
                                // maximum probability of the product, an prod_p_out is 0 in this
                                // case.
                                if prod_p_in <= FLOAT_CMP_THRESHOLD {
                                    is_node_sat = false;
                                    break;
                                }
                                child_entry.add_key(self.component_extractor[sub_component].get_cache_key());
                            }
                        }
                        if is_node_sat && prod_p_in > FLOAT_CMP_THRESHOLD {
                            children_map.insert(variable, child_entry);
                        }
                        self.restore();
                        prod_p_out = prod_maximum_probability.clone() - prod_p_out;
                        p_in += prod_p_in * &p;
                        p_out += prod_p_out * &p;
                    }
                }
                self.restore();
                child_id += 1;
            }
            let cache_entry = SearchCacheEntry::new((p_in, p_out),
                discrepancy,
                Some(distribution),
                children_map,
            );
            (cache_entry, level - 1)
        } else {
            (SearchCacheEntry::new((F128!(1.0), F128!(0.0)), usize::MAX, None, FxHashMap::default()), level - 1)
        }
    }
}

impl<B: BranchingDecision, const S: bool> Solver<B, S> {

    pub fn compile<R: SemiRing>(&mut self, is_lds: bool) -> Dac<R> {
        let start = Instant::now();
        self.state.save_state();
        let preproc_result = self.preprocess();
        // Create the DAC and add elements from the preprocessing
        let mut dac = Dac::new();
        let forced_by_propagation = self.forced_from_propagation();

        self.restructure_after_preprocess();

        if preproc_result.is_some() {
            return dac;
        }
        // Perform the actual search that will fill the cache
        let _ = self.pwmc(is_lds);

        // Adds the distributions in the circuit
        for distribution in self.problem.distributions_iter() {
            for v in self.problem[distribution].iter_variables() {
                let _ = dac.distribution_value_node(&self.problem, distribution, v);
            }
        }

        let mut has_node_search = false;
        let mut root_number_children = if self.cache.contains_key(&self.component_extractor[ComponentIndex(0)].get_cache_key()) { has_node_search = true; 1 } else { 0 };
        root_number_children += forced_by_propagation.0.len() + forced_by_propagation.1.len();
        let root = dac.prod_node(root_number_children);
        let mut child_id = root.input_start();

        // Forced variables from the propagation
        for (d, variable) in forced_by_propagation.0.iter().copied() {
            let distribution_node = dac.distribution_value_node(&self.problem, d, variable);
            dac.add_input(child_id, distribution_node);
            child_id += 1;
        }
        // Unconstrained distribution variables
        for (d, values) in forced_by_propagation.1 {
            let sum_node = dac.sum_node(values.len());
            for (i, v) in values.iter().copied().enumerate() {
                let distribution_node = dac.distribution_value_node(&self.problem, d, v);
                dac.add_input(sum_node.input_start() + i, distribution_node);
            }
            let sum_id = dac.add_node(sum_node);
            dac.add_input(child_id, sum_id);
            child_id += 1;
        }

        let mut map: FxHashMap<CacheKey, NodeIndex> = FxHashMap::default();
        if has_node_search {
            let node_search = self.build_dac(&mut dac, self.component_extractor[ComponentIndex(0)].get_cache_key(), &mut map);
            dac.add_input(child_id, node_search);
        }
        let root = dac.add_node(root);
        dac.set_root(root);

        dac.set_compile_time(start.elapsed().as_secs());
        dac
    }

    pub fn build_dac<R: SemiRing>(&self, dac: &mut Dac<R>, component_key: CacheKey, c: &mut FxHashMap<CacheKey, NodeIndex>) -> NodeIndex {
        if let Some(child_i) = c.get(&component_key) {
            return *child_i;
        }
        let current = self.cache.get(&component_key).unwrap();
        let sum_node_child = current.number_children();
        let sum_node = dac.sum_node(sum_node_child);

        let mut sum_node_child = 0;
        // Iterate on the variables the distribution with the associated cache key
        for variable in current.children_variables() {
            let number_children = current.variable_number_children(variable);
            if number_children == 0 {
                continue;
            }
            let prod_node = dac.prod_node(number_children);

            let mut child_id = prod_node.input_start();
            // Adding to the new product node all the propagated variables, including the distribution value we branch on
            for (d, v) in current.forced_choices(variable).iter().copied() {
                let distribution_prop = dac.distribution_value_node(&self.problem, d, v);
                dac.add_input(child_id, distribution_prop);
                child_id += 1;

            }

            // Adding to the new product node sum nodes for all the unconstrained distribution not summing to 1
            for (d, values) in current.unconstrained_distribution_variables_of(variable) {
                let sum_node_unconstrained = dac.sum_node(values.len());
                for (i, v) in values.iter().copied().enumerate() {
                    let distribution_unconstrained = dac.distribution_value_node( &self.problem, *d, v);
                    dac.add_input(sum_node_unconstrained.input_start() + i, distribution_unconstrained);
                }
                let id = dac.add_node(sum_node_unconstrained);
                dac.add_input(child_id, id);
                child_id += 1;
            }
            // Recursively build the DAC for each sub-component
            for cache_key in current.child_keys(variable) {
                let id = self.build_dac(dac, cache_key, c);
                dac.add_input(child_id, id);
                child_id += 1;
            }
            let id = dac.add_node(prod_node);
            dac.add_input(sum_node.input_start() + sum_node_child, id);
            sum_node_child += 1;
        }
        let sum_index = dac.add_node(sum_node);
        c.insert(component_key, sum_index);
        sum_index
    }

    /// Returns the choices (i.e., assignments to the distributions) made during the propagation as
    /// well as the distributions that are not constrained anymore.
    /// A choice for a distribution is a pair (d, i) = (DistributionIndex, usize) that indicates that
    /// the i-th value of disitribution d is true.
    /// An unconstrained distribution is a pair (d, v) = (DistributionIndex, Vec<usize>) that
    /// indicates that distribution d does not appear in any clauses and its values in v are not
    /// set yet.
    fn forced_from_propagation(&mut self) -> (Vec<DistributionChoice>, Vec<UnconstrainedDistribution>) {
        let mut forced_distribution_variables: Vec<DistributionChoice> = vec![];
        let mut unconstrained_distribution_variables: Vec<UnconstrainedDistribution> = vec![];

        if self.propagator.has_assignments(&self.state) || self.propagator.has_unconstrained_distribution() {
            // First, we look at the assignments
            for literal in self.propagator.assignments_iter(&self.state) {
                let variable = literal.to_variable();
                // Only take probabilistic variables set to true
                if self.problem[variable].is_probabilitic() && literal.is_positive() && self.problem[variable].weight().unwrap() != 1.0 {
                    let distribution = self.problem[variable].distribution().unwrap();
                    // This represent which "probability index" is send to the node
                    forced_distribution_variables.push((distribution, variable));
                }
            }

            // Then, for each unconstrained distribution, we create a sum_node, but only if the
            // distribution has at least one value set to false.
            // Otherwise it would always send 1.0 to the product node.
            for distribution in self.propagator.unconstrained_distributions_iter() {
                if self.problem[distribution].number_false(&self.state) != 0 {
                    let values = self.problem[distribution].iter_variables().filter(|v| !self.problem[*v].is_fixed(&self.state)).collect::<Vec<VariableIndex>>();
                    unconstrained_distribution_variables.push((distribution, values));
                }
            }
            
        }
        (forced_distribution_variables, unconstrained_distribution_variables)
    }
}

#[derive(Default, Clone)]
pub struct CacheChildren {
    children_keys: Vec<CacheKey>,
    forced_choices: Vec<DistributionChoice>,
    unconstrained_distributions: Vec<UnconstrainedDistribution>,
}

impl CacheChildren {
    pub fn new(forced_choices: Vec<DistributionChoice>, unconstrained_distributions: Vec<UnconstrainedDistribution>) -> Self {
        Self {
            children_keys: vec![],
            forced_choices,
            unconstrained_distributions,
        }
    }

    pub fn add_key(&mut self, key: CacheKey) {
        self.children_keys.push(key);
    }
}
/// An entry in the cache for the search. It contains the bounds computed when the sub-problem was
/// explored as well as various informations used by the solvers.
#[derive(Clone)]
pub struct SearchCacheEntry {
    /// The current bounds on the sub-problem
    bounds: Bounds,
    /// Maximum discrepancy used for that node
    discrepancy: usize,
    /// The distribution on which to branch in this problem
    distribution: Option<DistributionIndex>,
    children: FxHashMap<VariableIndex, CacheChildren>,
}

impl SearchCacheEntry {

    /// Returns a new cache entry
    pub fn new(bounds: Bounds, discrepancy: usize, distribution: Option<DistributionIndex>, children: FxHashMap<VariableIndex, CacheChildren>) -> Self {
        Self {
            bounds,
            discrepancy,
            distribution,
            children,
        }
    }

    /// Returns a reference to the bounds of this entry
    pub fn bounds(&self) -> Bounds {
        self.bounds.clone()
    }

    /// Returns the discrepancy of the node
    pub fn discrepancy(&self) -> usize {
        self.discrepancy
    }

    pub fn distribution(&self) -> Option<DistributionIndex> {
        self.distribution
    }

    pub fn forced_choices(&self, variable: VariableIndex) -> &Vec<DistributionChoice> {
        &self.children.get(&variable).unwrap().forced_choices
    }

    pub fn unconstrained_distribution_variables_of(&self, variable: VariableIndex) -> &Vec<UnconstrainedDistribution> {
        &self.children.get(&variable).unwrap().unconstrained_distributions
    }

    pub fn number_children(&self) -> usize {
        self.children.len()
    }

    pub fn variable_number_children(&self, variable: VariableIndex) -> usize {
        let entry = self.children.get(&variable).unwrap();
        entry.children_keys.len() + entry.forced_choices.len() + entry.unconstrained_distributions.len()
    }

    pub fn children_variables(&self) -> Vec<VariableIndex> {
        self.children.keys().copied().collect()
    }

    pub fn child_keys(&self, variable: VariableIndex) -> Vec<CacheKey> {
        self.children.get(&variable).unwrap().children_keys.clone()
    }
}
