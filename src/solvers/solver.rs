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

use super::statistics::Statistics;
use super::*;
use crate::branching::BranchingDecision;
use crate::common::*;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::problem::{DistributionIndex, Problem, VariableIndex};
use crate::diagrams::dac::dac::Dac;
use crate::diagrams::semiring::SemiRing;
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use crate::PEAK_ALLOC;
use rug::Float;
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
        self.start = Instant::now();
        if let Some(sol) = self.preprocess() {
            return sol;
        }
        if !is_lds {
            let sol = self.do_discrepancy_iteration(usize::MAX);
            println!("Solution: {}", sol);
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

    /// Preprocess the problem, if the problem is solved during the preprocess, return a solution.
    /// Returns None otherwise
    fn preprocess(&mut self) -> Option<Solution> {
        self.propagator.init(self.problem.number_clauses());
        self.state.save_state();
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
        self.preproc_out = Some(
            1.0 - self
                .problem
                .distributions_iter()
                .map(|d| self.problem[d].remaining(&self.state))
                .product::<f64>(),
        );
        self.problem.clear_after_preprocess(&mut self.state);
        self.state.restore_state();
        if self.problem.number_clauses() == 0 {
            let lb = self.preproc_in.clone().unwrap();
            let ub = F128!(1.0 - self.preproc_out.unwrap());
            return Some(Solution::new(lb, ub, self.start.elapsed().as_secs()));
        }
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
        self.propagator.set_forced();
        None
    }

    pub fn compile<R: SemiRing>(&mut self) -> Dac<R> { //discrepancy: Option<usize>
        // TODO discrepancy: 
        // do we want a compilation timeout? then compile a the end of each discrepancy iteration
        // or do we want to give a specific discrepancy to achieve? Then adapt search to immediately use that value
        let start = Instant::now();
        let discrepancy: Option<usize> = None;
        self.search(discrepancy.is_some());
        let mut dac = Dac::new();
        let root_component = ComponentIndex(0);
        let parent_i = dac.add_prod_node();
        dac.set_root(parent_i);
        let mut map: FxHashMap<CacheKey, NodeIndex> = FxHashMap::default();
        if let Some(root_c) = self.cache.get(&self.component_extractor[root_component].get_cache_key()) {
            dac[parent_i].set_bounds(root_c.bounds().clone());
        }
        self.build_dac(&mut dac, parent_i, self.component_extractor[root_component].get_cache_key(), &mut map);
        dac.optimize_structure();
        dac.set_compile_time(start.elapsed().as_secs());
        dac
    }

    pub fn build_dac<R: SemiRing>(&self, dac: &mut Dac<R>, parent_node: NodeIndex, component_key: CacheKey, c: &mut FxHashMap<CacheKey, NodeIndex>) {
        let do_print = false;
        if do_print{ println!("*************Parent node: {:?}", parent_node);}

        if let Some(child_i) = c.get_mut(&component_key) {
                dac.add_node_output(*child_i, parent_node);
                return;
        }
        if let Some(current) = self.cache.get(&component_key) {
            if current.bounds().0 == 0.0 {
                println!("zero bound");
                return;
            }
            // Calcul de epsilon sur base de upper et lower?
            if let Some(distribution) = current.distribution() {
                if do_print{ println!("Distribution: {:?}", distribution);}
                let sum_node = dac.add_sum_node();
                dac[sum_node].set_bounds(current.bounds().clone());
                c.insert(component_key, sum_node);
                dac.add_node_output(sum_node, parent_node);
                // Iterate on the variables the distribution with the associated cache key
                for (variable, keys) in current.variable_component_keys(){
                    if distribution.0 == 8 && variable == 0 {
                        println!("8,0");
                    }
                    let mut is_unsat = true;
                    // Create a new product node for the distribution value
                    let prod_node_i = dac.add_prod_node();
                    let distribution_i = dac.distribution_value_node_index(distribution, variable, self.problem[VariableIndex(variable+ self.problem[distribution].start().0)].weight().unwrap());
                    if do_print{ println!("Distribution value node: {:?} with proba {}", distribution_i, self.problem[VariableIndex(variable+ self.problem[distribution].start().0)].weight().unwrap());}
                    dac.add_node_output(distribution_i, prod_node_i);

                    // Adding to the new product node all the propagated variables
                    if let Some(forced_distribution_variables) = current.forced_distribution_variables_of(variable) {
                        for (d, v) in forced_distribution_variables {
                            let distribution_prop = dac.distribution_value_node_index(d, v, self.problem[VariableIndex(v + self.problem[d].start().0)].weight().unwrap());
                            if do_print{ println!("Forced distribution value node: {:?} with proba {}", distribution_prop, self.problem[VariableIndex(v + self.problem[d].start().0)].weight().unwrap());}
                            dac.add_node_output(distribution_prop, prod_node_i);
                        }
                        is_unsat = false;
                    }

                    // Adding to the new product node sum nodes for all the unconstrained distribution not summing to 1
                    if let Some(unconstrained_distribution_variables) = current.unconstrained_distribution_variables_of(variable) {
                        for (d, values) in unconstrained_distribution_variables {
                            let sum_node_unconstrained = dac.add_sum_node();
                            for v in values {
                                let distribution_unconstrained = dac.distribution_value_node_index(d, v, self.problem[VariableIndex(v + self.problem[d].start().0)].weight().unwrap());
                                if do_print{ println!("Unconstrained distribution value node: {:?} with proba {}", distribution_unconstrained, self.problem[VariableIndex(v + self.problem[d].start().0)].weight().unwrap());}
                                dac.add_node_output(distribution_unconstrained, sum_node_unconstrained);
                            }
                            dac.add_node_output(sum_node_unconstrained, prod_node_i);
                        }
                        is_unsat = false;
                    }
                    if !is_unsat {
                        dac.add_node_output(prod_node_i, sum_node);
                        // Recursively build the DAC for each sub-component
                        for cache_key in keys {
                            if distribution.0 == 8 && variable == 0 {
                                println!("Keys");
                            }
                            self.build_dac(dac, prod_node_i, cache_key.clone(), c);
                        }
                    }                    
                }
            }    
        }
    }

    pub fn do_discrepancy_iteration(&mut self, discrepancy: usize) -> Solution {
        let (result,_) = self.get_bounds_from_cache(ComponentIndex(0), (1.0+self.epsilon).powf(2.0), 1, discrepancy);
        //self.solve_components(ComponentIndex(0),1, (1.0 + self.epsilon).powf(2.0), discrepancy);
        let p_in = result.bounds().0.clone();
        let p_out = result.bounds().1.clone();
        let lb = p_in * self.preproc_in.clone().unwrap();
        let ub: Float =
            1.0 - (self.preproc_out.unwrap() + p_out * self.preproc_in.clone().unwrap());
        Solution::new(lb, ub, self.start.elapsed().as_secs())
    }

    /// Split the component into multiple sub-components and solve each of them
    /* fn solve_components(&mut self, component: ComponentIndex, level: isize, bound_factor: f64, discrepancy: usize) -> (Bounds, isize) {
        if PEAK_ALLOC.current_usage_as_mb() as u64 >= self.mlimit {
            self.cache.clear();
        }
        self.state.save_state();
        let mut p_in = F128!(1.0);
        let mut p_out = F128!(1.0);
        // Computes the maximum probability of the sub-problem being solved. This is used for the
        // upper bound computation.
        let mut maximum_probability = F128!(1.0);
        for distribution in self
            .component_extractor
            .component_distribution_iter(component)
            .filter(|d| self.problem[*d].is_constrained(&self.state)) {
            maximum_probability *= self.problem[distribution].remaining(&self.state);
        }

        // If there are no more component to explore (i.e. the sub-problem only contains
        // deterministic variables), then detect_components return false.
        if self.component_extractor.detect_components(
            &mut self.problem,
            &mut self.state,
            component,
            &mut self.propagator,
        ) {
            self.statistics.and_node();
            let number_components = self.component_extractor.number_components(&self.state);
            self.statistics.decomposition(number_components);
            let new_bound_factor = bound_factor.powf(1.0 / number_components as f64);
            for sub_component in self.component_extractor.components_iter(&self.state) {
                // If there are no more time and that all components could not be explored, we
                // return 0 for both p_in and p_out; it is the same result as if no component were
                // explored.
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
                let (sub_p_in, sub_p_out) = sub_problem.bounds();
                p_in *= sub_p_in;
                p_out *= sub_maximum_probability - sub_p_out.clone();
            }
        }
        self.restore();
        ((p_in, maximum_probability - p_out), level - 1) //Renvoyer aussi une liste des subcomponents id? pour mettre comme info dans la cacheEntry?
    }
    */
    
    /// Retrieves the bounds of a sub-problem from the cache. If the sub-problem has never been
    /// explored or that the bounds, given the bounding factor, are not good enough, the
    /// sub-problem is solved and the result is inserted in the cache.
    fn get_bounds_from_cache(
        &mut self,
        component: ComponentIndex,
        bound_factor: f64,
        level: isize,
        discrepancy: usize,
    ) -> SearchResult {
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

    fn forced_from_propagation(&mut self, branched_on: VariableIndex) -> (Vec<(DistributionIndex, usize)>, Vec<(DistributionIndex, Vec<usize>)>) {
        let mut forced_distribution_variables: Vec<(DistributionIndex, usize)> = vec![];
        let mut unconstrained_distribution_variables: Vec<(DistributionIndex, Vec<usize>)> = vec![];

        if self.propagator.has_assignments(&self.state) || self.propagator.has_unconstrained_distribution() {
            //let node = dac.add_prod_node();
            // First, we look at the assignments
            for literal in self.propagator.assignments_iter(&self.state) {
                let variable = literal.to_variable();
                // Only take probabilistic variables set to true
                if self.problem[variable].is_probabilitic() && literal.is_positive() && variable != branched_on {
                    let distribution = self.problem[variable].distribution().unwrap();
                    // This represent which "probability index" is send to the node
                    let value_index = variable.0 - self.problem[distribution].start().0;
                    forced_distribution_variables.push((distribution, value_index));
                    //let distribution_node = dac.distribution_value_node_index(distribution, value_index, self.problem[variable].weight().unwrap());
                    //dac.add_node_output(distribution_node, node);
                }
            }

            // Then, for each unconstrained distribution, we create a sum_node, but only if the
            // distribution has at least one value set to false.
            // Otherwise it would always send 1.0 to the product node.
            for distribution in self.propagator.unconstrained_distributions_iter() {
                if self.problem[distribution].number_false(&self.state) != 0 {
                    //let sum_node = dac.add_sum_node();
                    let mut values = vec![];
                    for variable in self.problem[distribution].iter_variables() {
                        if !self.problem[variable].is_fixed(&self.state) {
                            let value_index = variable.0 - self.problem[distribution].start().0;
                            //let distribution_node = dac.distribution_value_node_index(distribution, value_index, self.problem[variable].weight().unwrap());
                            //dac.add_node_output(distribution_node, sum_node);
                            values.push(value_index);
                        }
                    }
                    unconstrained_distribution_variables.push((distribution, values));
                    //dac.add_node_output(sum_node, node);
                }
            }
            
        }
        (forced_distribution_variables, unconstrained_distribution_variables)
    }

    /// Choose a distribution on which to branch, in the sub-problem, and solves the sub-problems
    /// resulting from the branching, recursively.
    /// Returns the bounds of the sub-problem as well as the level to which the solver must
    /// backtrack.
    fn branch(
        &mut self,
        component: ComponentIndex,
        level: isize,
        bound_factor: f64,
        discrepancy: usize,
        choice: Option<DistributionIndex>,
    ) -> SearchResult {
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
            let mut variable_component_keys: Vec<(usize, Vec<CacheKey>)> = vec![];
            let mut forced_distribution_variables: Vec<(usize, Vec<(DistributionIndex, usize)>)> = vec![];
            let mut unconstrained_distribution_variables: Vec<(usize, Vec<(DistributionIndex, Vec<usize>)>)> = vec![];
            for variable in self.problem[distribution].iter_variables() {
                if distribution.0 == 8 {
                    println!("Variable: {}", variable.0- self.problem[distribution].start().0);
                }
                let mut keys = vec![];

                if self.problem[variable].is_fixed(&self.state) {
                    continue;
                }
                if self.start.elapsed().as_secs() >= self.timeout || child_id == discrepancy {
                    break;
                }
                let v_weight = self.problem[variable].weight().unwrap();
                self.state.save_state();
                match self.propagator.propagate_variable(
                    variable,
                    true,
                    &mut self.problem,
                    &mut self.state,
                    component,
                    &mut self.component_extractor,
                    level,
                ) {
                    Err(backtrack_level) => {
                        if distribution.0 == 8 && variable.0 - self.problem[distribution].start().0 == 0 {
                            println!("unsat");
                        }
                        self.statistics.unsat();
                        // The assignment triggered an UNSAT, so the whole sub-problem is part of
                        // the non-models.
                        p_out += v_weight * unsat_factor;
                        //forced_distribution_variables.push((variable.0 - self.problem[distribution].start().0, vec![]));
                        //unconstrained_distribution_variables.push((variable.0 - self.problem[distribution].start().0, vec![]));
                        if backtrack_level != level {
                            // The clause learning scheme tells us that we need to backtrack
                            // non-chronologically. There are no models in this sub-problem
                            self.restore();
                            println!("distribution: {:?}, variable: {}", distribution, variable.0 - self.problem[distribution].start().0);
                            println!("out1");
                            return (SearchCacheEntry::new((F128!(0.0), F128!(maximum_probability)), usize::MAX, Some(distribution), vec![], vec![], vec![]), backtrack_level);
                        }
                    }
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
                        // It is possible that the propagation removes enough variable so that the
                        // bounds are close enough
                        if p != 0.0 {
                            let new_discrepancy = discrepancy - child_id;
                            //let ((child_p_in, child_p_out), backtrack_level) = self.solve_components(component, level + 1, bound_factor, new_discrepancy);
                            let mut child_p_in = F128!(1.0);
                            let mut child_p_out = F128!(1.0);
                            let mut child_maximum_probability = F128!(1.0);
                            
                            let (forced_distribution_var, unconstrained_distribution_var) = self.forced_from_propagation(variable);
                            forced_distribution_variables.push((variable.0 - self.problem[distribution].start().0, forced_distribution_var));
                            unconstrained_distribution_variables.push((variable.0 - self.problem[distribution].start().0, unconstrained_distribution_var));
                            // *****************************
                            // Old Solve_components function
                            if PEAK_ALLOC.current_usage_as_mb() as u64 >= self.mlimit {
                                self.cache.clear();
                            }
                            self.state.save_state();
                            for child_distribution in self.component_extractor.component_distribution_iter(component) {
                                if self.problem[child_distribution].is_constrained(&self.state) {
                                    child_maximum_probability *= self.problem[child_distribution].remaining(&self.state);
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
                                    keys.push(self.component_extractor[sub_component].get_cache_key());
                                    if self.start.elapsed().as_secs() >= self.timeout {
                                        //return ((F128!(0.0), F128!(0.0)), level - 1);
                                        return (SearchCacheEntry::new((p_in, p_out), usize::MAX, Some(distribution), variable_component_keys, forced_distribution_variables, unconstrained_distribution_variables), level - 1);
                                    }
                                    let sub_maximum_probability = self.component_extractor[sub_component].max_probability();
                                    assert!(0.0 <= sub_maximum_probability && sub_maximum_probability <= 1.0);
                                    let (sub_problem, backtrack_level) = self.get_bounds_from_cache(sub_component, new_bound_factor, level +1, new_discrepancy); // the function was called with level + 1 and level was given
                                    if backtrack_level != level { // the function was called with level + 1 and level was used here
                                        self.restore();
                                        // backtrack_level = backtrack_level;
                                        //return ((F128!(0.0), maximum_probability), backtrack_level);
                                        return (SearchCacheEntry::new((F128!(0.0), F128!(maximum_probability)), usize::MAX, Some(distribution), variable_component_keys, forced_distribution_variables, unconstrained_distribution_variables), backtrack_level);
                                    }
                                    // If any of the component is not fully explored, then so is the node
                                    let (sub_p_in, sub_p_out) = sub_problem.bounds();
                                    child_p_in *= sub_p_in;
                                    child_p_out *= sub_maximum_probability - sub_p_out.clone();
                                }
                            }
                            self.restore();
                            // child_p_in = child_p_in;
                            child_p_out = child_maximum_probability - child_p_out;
                            //((p_in, maximum_probability - p_out), level - 1)
                            
                            
                            
                            
                            // ****************************************
                            // Return to the end of the branch function
                            /* if backtrack_level != level {
                                self.restore();
                                println!("quit");
                                return (SearchCacheEntry::new((F128!(0.0), F128!(maximum_probability)), usize::MAX, Some(distribution), variable_component_keys, forced_distribution_variables, unconstrained_distribution_variables), backtrack_level);
                            } */
                            p_in += child_p_in * &p;
                            p_out += child_p_out * &p;
                        }
                    }
                }
                variable_component_keys.push((variable.0-self.problem[distribution].start().0, keys));
                self.restore();
                child_id += 1;
            }
            let cache_entry = SearchCacheEntry::new((p_in, p_out), discrepancy, Some(distribution), variable_component_keys, forced_distribution_variables, unconstrained_distribution_variables);
            (cache_entry, level - 1)
        } else {
            println!("out2");
            (SearchCacheEntry::new((F128!(1.0), F128!(0.0)), usize::MAX, None, vec![], vec![], vec![]), level - 1)
        }
    }
}
