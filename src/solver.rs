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
    cache: FxHashMap<CacheKey, CacheEntry>,
    /// Statistics gathered during the solving
    statistics: Statistics<S>,
    /// Product of the weight of the variables set to true during propagation
    preproc_in: Option<Float>,
    /// Probability of removed interpretation during propagation
    preproc_out: Option<f64>,
    /// Parameters of the solving
    parameters: SolverParameters,
    cache_keys: Vec<CacheKey>,
}

impl<B: BranchingDecision, const S: bool> Solver<B, S> {
    pub fn new(
        problem: Problem,
        state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: Box<B>,
        propagator: Propagator,
        parameters: SolverParameters,
    ) -> Self {
        Self {
            problem,
            state,
            component_extractor,
            branching_heuristic,
            propagator,
            cache: FxHashMap::default(),
            statistics: Statistics::default(),
            preproc_in: None,
            preproc_out: None,
            parameters,
            cache_keys: vec![],
        }
    }

    /// Restores the state of the solver to the previous state
    fn restore(&mut self) {
        self.propagator.restore(&self.state);
        self.state.restore_state();
    }

    /// Solves the problem represented by this solver using a DPLL-search based method.
    pub fn search(&mut self, is_lds: bool) -> Solution {
        self.parameters.start = Instant::now();
        self.state.save_state();
        if let Some(sol) = self.preprocess() {
            return sol;
        }
        self.restructure_after_preprocess();
        if self.problem.number_clauses() == 0 {
            let lb = self.preproc_in.clone().unwrap();
            let ub = F128!(1.0 - self.preproc_out.unwrap());
            return Solution::new(lb, ub, self.parameters.start.elapsed().as_secs(), None);
        }
        if !is_lds {
            let sol = self.do_discrepancy_iteration(usize::MAX);
            self.statistics.print();
            sol
        } else {
            let mut discrepancy = 1;
            let mut complete_sol = None;
            loop {
                let solution = self.do_discrepancy_iteration(discrepancy);
                if self.parameters.start.elapsed().as_secs() < self.parameters.timeout || complete_sol.as_ref().is_none() {
                    solution.print();
                    complete_sol = Some(solution);
                }
                if self.parameters.start.elapsed().as_secs() >= self.parameters.timeout || complete_sol.as_ref().unwrap().has_converged(self.parameters.epsilon) {
                    self.statistics.print();
                    return complete_sol.unwrap()
                }
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
                self.parameters.start.elapsed().as_secs(),
                None,
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
        let remaining = self.problem.distributions_iter().map(|d| {
            self.problem[d].remaining(&self.state)
        }).collect::<Vec<f64>>();
        self.state.restore_state();
        for (i, distribution) in self.problem.distributions_iter().enumerate() {
            self.problem[distribution].set_remaining(remaining[i], &mut self.state);
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
    }

    pub fn do_discrepancy_iteration(&mut self, discrepancy: usize) -> Solution {
        let result = self.pwmc(ComponentIndex(0), 1, discrepancy, discrepancy);
        let p_in = result.bounds.0.clone();
        let p_out = result.bounds.1.clone();
        let lb = p_in * self.preproc_in.clone().unwrap();
        let ub: Float = 1.0 - (self.preproc_out.unwrap() + p_out * self.preproc_in.clone().unwrap());
        let cache_index = result.cache_index;
        Solution::new(lb, ub, self.parameters.start.elapsed().as_secs(), Some(cache_index))
    }

    fn pwmc(&mut self, component: ComponentIndex, level: isize, discrepancy: usize, init_d:usize) -> SearchResult {
        let search = false;
        let do_ub = true;
        if PEAK_ALLOC.current_usage_as_mb() as u64 >= self.parameters.memory_limit {
            self.cache.clear();
        }
        let cache_key = self.component_extractor[component].get_cache_key();
        self.statistics.cache_access();

        let mut backtrack_level = level - 1;
        let maximum_probability = self.component_extractor[component].max_probability();

        let mut all_component_d_values = vec![];
        if !search && do_ub {
            // For upper bound, collect all the distributions' variables that are not fixed, and ignoring the distribution still summing to 1
            // This information is used for upper-bound AC compilation when decomposing in independant subcomponents as: remaining problem - contribution of each subcomponent
            let all_component_distributions = self.component_extractor.component_distribution_iter(component).filter(|d| self.problem[*d].is_constrained(&self.state)).collect::<Vec<DistributionIndex>>();
            let all_component_d_v = all_component_distributions.iter().map(|d| (*d, self.problem[*d].iter_variables().filter(|v| !self.problem[*v].is_fixed(&self.state)).collect::<Vec<VariableIndex>>())).collect::<Vec<(DistributionIndex, Vec<VariableIndex>)>>();
            let nonempty = all_component_d_v[0].clone();
            for (d, v) in all_component_d_v.iter() {
                let sum = v.iter().map(|v| self.problem[*v].weight().unwrap()).sum::<f64>();
                if sum < 1.0 { 
                    all_component_d_values.push((d.clone(), v.clone())); 
                }
            }
            if all_component_d_values.is_empty() { all_component_d_values.push(nonempty); }
        }
        // Retrieve the cache entry if it exists, otherwise create a new one
        let mut cache_entry = self.cache.remove(&cache_key).unwrap_or_else(|| {
            self.statistics.cache_miss();
            let cache_key_index = self.cache_keys.len();
            if !search {
                self.cache_keys.push(cache_key.clone());
            }
            self.statistics.or_node();
            let distribution = self.branching_heuristic.branch_on(&self.problem, &mut self.state, &self.component_extractor, component);
            CacheEntry::new((F128!(0.0), F128!(0.0)), 0, distribution, FxHashMap::default(), cache_key_index, FxHashMap::default(), maximum_probability, all_component_d_values, init_d-discrepancy, FxHashMap::default())
            //CacheEntry::new((F128!(0.0), F128!(0.0)), 0, None, FxHashMap::default(), cache_key_index, FxHashMap::default(), maximum_probability, all_component_d_values, init_d-discrepancy, FxHashMap::default())
        });
        /* if cache_entry.distribution.is_none() {
            self.statistics.or_node();
            cache_entry.distribution = self.branching_heuristic.branch_on(&self.problem, &mut self.state, &self.component_extractor, component);
        } */

        if cache_entry.discrepancy < discrepancy && !cache_entry.are_bounds_tight(maximum_probability) {
            let mut new_p_in = F128!(0.0);
            let mut new_p_out = F128!(0.0);
            let distribution = cache_entry.distribution.unwrap();
            let unsat_factor = maximum_probability / self.problem[distribution].remaining(&self.state);
            let mut unsat_d_values = vec![];
            if !search && do_ub {
                // For upper-bound, collect all the distributions' variables that are not fixed, excluding the current distribution that will be branched on, and ignoring the distribution still summing to 1
                // This information is used for upper-bound AC compilation when reaching unsat in a independant component, to account for all interpretations concerned by the unsat
                let unsat_distributions = self.component_extractor.component_distribution_iter(component).filter(|d| self.problem[*d].is_constrained(&self.state)&&*d!=distribution).collect::<Vec<DistributionIndex>>();
                let unsat_d_v = unsat_distributions.iter().map(|d| (*d, self.problem[*d].iter_variables().filter(|v| !self.problem[*v].is_fixed(&self.state)).collect::<Vec<VariableIndex>>())).collect::<Vec<(DistributionIndex, Vec<VariableIndex>)>>();
                let nonempty = unsat_d_v[0].clone();
                for (d, v) in unsat_d_v.iter() {
                    let sum = v.iter().map(|v| self.problem[*v].weight().unwrap()).sum::<f64>();
                    if sum < 1.0 { unsat_d_values.push((d.clone(), v.clone())); }
                }
                if unsat_d_values.is_empty() { unsat_d_values.push(nonempty); }
            }
            
            let mut child_id = 0;
            for variable in self.problem[distribution].iter_variables() {
                if self.problem[variable].is_fixed(&self.state) {
                    continue;
                }
                if self.parameters.start.elapsed().as_secs() >= self.parameters.timeout || child_id == discrepancy {
                    break;
                }
                if cache_entry.child_discrepancy.get(&variable).is_none() {
                    cache_entry.child_discrepancy.insert(variable, (init_d-discrepancy)+child_id);
                }
                let v_weight = self.problem[variable].weight().unwrap();
                self.state.save_state();
                match self.propagator.propagate_variable(variable, true, &mut self.problem, &mut self.state, component, &mut self.component_extractor, level) {
                    Err(bt) => {
                        // The propagation is unsat
                        self.statistics.unsat();
                        new_p_out += v_weight * unsat_factor;
                        if bt != level {
                            self.restore();
                            backtrack_level = bt;
                            cache_entry.bounds.1 = new_p_out.clone();
                            new_p_out = F128!(maximum_probability);
                            cache_entry.clear_children();
                            break;
                        }
                        let mut component_distributions = vec![];
                        if !search && do_ub {
                            component_distributions = self.component_extractor.component_distribution_iter(component).collect::<Vec<DistributionIndex>>();
                        }
                        self.restore(); // Restore the state before the propagation
                        // Retrieve all the distributions' variables that are not fixed when branching to the unsat, and ignoring the distribution still summing to 1
                        let mut excluded_distribtions = vec![];
                        if !search && do_ub {
                            for d in component_distributions.iter() {
                                let mut variables = vec![];
                                let mut total = 0.0;
                                if *d != distribution {
                                    for v in self.problem[*d].iter_variables() {
                                        if !self.problem[v].is_fixed(&self.state) {
                                            variables.push(v);
                                            total += self.problem[v].weight().unwrap();
                                        }
                                    }
                                    if (!variables.is_empty() && total < 1.0) || excluded_distribtions.is_empty() {
                                        excluded_distribtions.push((*d, variables));
                                    }
                                }
                            }
                            let child_entry = CacheChildren::new(vec![(distribution, variable)], vec![], excluded_distribtions);
                            cache_entry.unsat_children.insert(variable, child_entry);
                        }
                    },
                    Ok(_) => {
                        let p = self.propagator.get_propagation_prob();
                        let removed = unsat_factor - self.component_extractor
                            .component_distribution_iter(component)
                            .filter(|d| *d != distribution)
                            .map(|d| self.problem[d].remaining(&self.state))
                            .product::<f64>();
                        
                        let mut remaining_d_values = vec![];
                        if !search && do_ub {
                            let remaining_distributions = self.component_extractor.component_distribution_iter(component).filter(|d| *d != distribution && self.problem[*d].is_constrained(&self.state)).collect::<Vec<DistributionIndex>>();
                            let remaining_d_v = remaining_distributions.iter().map(|d| (*d, self.problem[*d].iter_variables().filter(|v| !self.problem[*v].is_fixed(&self.state)).collect::<Vec<VariableIndex>>())).collect::<Vec<(DistributionIndex, Vec<VariableIndex>)>>();
                            for (d, v) in remaining_d_v.iter() {
                                let sum = v.iter().map(|v| self.problem[*v].weight().unwrap()).sum::<f64>();
                                if remaining_d_values.is_empty() || sum < 1.0 { remaining_d_values.push((d.clone(), v.clone())); }
                            }
                        }
                        
                        new_p_out += removed * v_weight;

                        // Decomposing into independent components
                        let mut prod_p_in = F128!(1.0);
                        let mut prod_p_out = F128!(1.0);
                        let prod_maximum_probability = self.component_extractor
                            .component_distribution_iter(component)
                            .filter(|d| self.problem[*d].is_constrained(&self.state))
                            .map(|d| self.problem[d].remaining(&self.state))
                            .product::<f64>();
                        let (forced_distribution_var, unconstrained_distribution_var, _) = self.forced_from_propagation(distribution, true);
                        let mut child_entry = CacheChildren::new(forced_distribution_var, unconstrained_distribution_var, vec![]);
                        if !search{
                            child_entry.total_distributions = unsat_d_values.clone();
                            child_entry.total_remaining_to_remove = remaining_d_values.clone();
                        }
                        self.state.save_state();
                        let mut is_product_sat = true;
                        let mut sat_keys = vec![];
                        let mut unsat_key= vec![];
                        if self.component_extractor.detect_components(&mut self.problem, &mut self.state, component) {
                            self.statistics.and_node();
                            self.statistics.decomposition(self.component_extractor.number_components(&self.state));
                            for sub_component in self.component_extractor.components_iter(&self.state) {
                                let sub_maximum_probability = self.component_extractor[sub_component].max_probability();
                                let sub_solution = self.pwmc(sub_component, level + 1, discrepancy - child_id, init_d);
                                prod_p_in *= &sub_solution.bounds.0;
                                prod_p_out *= sub_maximum_probability - sub_solution.bounds.1.clone();
                                if prod_p_in == 0.0 {
                                    is_product_sat = false;
                                    if !search {
                                        unsat_key.push(sub_solution.cache_index);
                                    }
                                    /* if prnt{ println!("prod p in {}, prod p out {}, unsat key {:?}", prod_p_in, prod_p_out, unsat_key);}
                                    if prod_p_out == 0.0 { 
                                        cache_entry.clear_children();
                                    } */
                                    backtrack_level = sub_solution.backtrack_level;
                                    if !search {
                                        cache_entry.bounds.1 = F128!(maximum_probability);
                                        cache_entry.bounds.0 = F128!(0.0);
                                        child_entry.excluded_distributions = unsat_d_values.clone();
                                    }
                                    break;
                                }
                                if !search {
                                    sat_keys.push(sub_solution.cache_index);
                                }
                            }
                        }
                        if is_product_sat {
                            if !search {
                                for key in sat_keys {
                                    child_entry.add_key(key);
                                }
                            }
                            cache_entry.children.insert(variable, child_entry);
                            if !search && cache_entry.unsat_children.contains_key(&variable) {
                                cache_entry.unsat_children.remove(&variable);
                            }
                        }
                        else if !search && !is_product_sat {
                            for k in unsat_key{
                                child_entry.add_key(k);
                            }
                            cache_entry.unsat_children.insert(variable, child_entry);
                            if cache_entry.children.contains_key(&variable) {
                                cache_entry.children.remove(&variable);
                            }
                        }
                        prod_p_out = prod_maximum_probability - prod_p_out;
                        new_p_in += prod_p_in * &p;
                        new_p_out += prod_p_out * &p;
                        self.restore(); // Restore the state before the decomposition
                        self.restore(); // Restore the state before the propagation
                    },
                }
                child_id += 1;
            }
            cache_entry.discrepancy = discrepancy;
            cache_entry.bounds = (new_p_in.clone(), new_p_out.clone());
        }
        /*else if discrepancy <= cache_entry.discrepancy && !cache_entry.are_bounds_tight(maximum_probability) {
            cache_entry.bounds.1 = F128!(1.0-maximum_probability);
            cache_entry.bounds.0 = F128!(0.0);
        }*/
        let result = SearchResult {
            bounds: cache_entry.bounds.clone(),
            backtrack_level,
            cache_index: cache_entry.cache_key_index,
        };
        if !search {
            self.cache.insert(cache_key, cache_entry);
        }
        else {
            self.cache.insert(cache_key, cache_entry);
        }
        result
    }
}

impl<B: BranchingDecision, const S: bool> Solver<B, S> {

    pub fn compile<R: SemiRing>(&mut self, is_lds: bool, lds_opti: bool) -> (Dac<R>, Dac<R>) {
        let start = Instant::now();
        self.state.save_state();

        let preproc_result = self.preprocess();
        // Create the DAC and add elements from the preprocessing
        let forced_by_propagation = self.forced_from_propagation(DistributionIndex(usize::MAX), true);
        // Remove the distributions' values from the excluded ones that had a weight of 0 or 1 and keep the old indexes of the others to be able to add them to the dac correctly
        let mut clean_excluded = vec![];
        let mut remaining = vec![];
        for (d, values) in forced_by_propagation.2.iter() {
            let mut new_val = vec![];
            for v in values.iter() {
                if self.problem[*v].weight().unwrap() != 1.0 && self.problem[*v].weight().unwrap() != 0.0 {
                    new_val.push((self.problem[*v].index_in_distribution().unwrap(), self.problem[*v].weight().unwrap()));
                }
            }
            if new_val.len() > 0 {
                let mut rem_val = vec![];
                for val in self.problem[*d].iter_variables() {
                    if !new_val.iter().any(|(v, _)| *v == self.problem[val].index_in_distribution().unwrap()) {
                        rem_val.push((self.problem[val].index_in_distribution().unwrap(), self.problem[val].weight().unwrap()));
                    }
                }
                clean_excluded.push((self.problem[*d].old_index(), new_val));
                remaining.push((self.problem[*d].old_index(), rem_val));
            }
        }
        // Similarly, keep the old indexes of the forced distributions' values to be able to add them to the dac correctly
        let mut clean_forced = vec![];
        for (d, v) in forced_by_propagation.0.iter() {
            clean_forced.push((self.problem[*d].old_index(), (self.problem[*v].index_in_distribution().unwrap(), self.problem[*v].weight().unwrap())));
        }
        // In the same way, keep the old indexes of the unconstrained distributions' values to be able to add them to the dac correctly
        let mut clean_unconstrained = vec![];
        for (d, values) in forced_by_propagation.1.iter() {
            let mut new_val = vec![];
            for v in values.iter() {
                new_val.push((self.problem[*v].index_in_distribution().unwrap(), self.problem[*v].weight().unwrap()));
            }
            clean_unconstrained.push((self.problem[*d].old_index(), new_val));
        }
        let forced_by_propagation = (clean_forced, clean_unconstrained, clean_excluded, remaining);
        self.restructure_after_preprocess();

        if preproc_result.is_some() {
            println!("to verifyyyy");
            return (Dac::default(), Dac::default());
        }
        // Perform the actual search that will fill the cache
        if self.problem.number_clauses() == 0 {
            let mut ac_model = self.build_ac(0.0, &forced_by_propagation, true, None, usize::MAX);
            let mut ac_nonmodel = self.build_ac(0.0, &forced_by_propagation, false, None, usize::MAX);
            ac_model.set_compile_time(start.elapsed().as_secs());
            ac_nonmodel.set_compile_time(start.elapsed().as_secs());
            return (ac_model, ac_nonmodel);
        }
        if !is_lds {
            let sol = self.do_discrepancy_iteration(usize::MAX);
            self.statistics.print();
            if sol.has_converged(0.0) && sol.bounds().0 < FLOAT_CMP_THRESHOLD {
                let mut ac_model = Dac::default();
                let mut ac_nonmodel = Dac::default();
                let n = ac_nonmodel.sum_node(0);
                let n_i = ac_nonmodel.add_node(n);
                ac_nonmodel.set_root(n_i);
                ac_model.set_compile_time(start.elapsed().as_secs());
                ac_nonmodel.set_compile_time(start.elapsed().as_secs());
                return (ac_model, ac_nonmodel);
            }
            let mut ac_model: Dac<R> = self.build_ac(sol.epsilon(), &forced_by_propagation, true, sol.cache_index(), usize::MAX);
            ac_model.set_compile_time(start.elapsed().as_secs());
            let mut ac_nonmodel: Dac<R> = self.build_ac(sol.epsilon(), &forced_by_propagation, false, sol.cache_index(), usize::MAX);
            ac_nonmodel.set_compile_time(start.elapsed().as_secs());
            println!("solution {:?} model out {}, nonmodel out 1-{} = {}", sol.bounds(), ac_model.evaluate().to_f64(), ac_nonmodel.evaluate().to_f64(), 1.0-ac_nonmodel.evaluate().to_f64());
            (ac_model, ac_nonmodel)
        } else {
            let mut discrepancy = 1;
            let mut complete_sol = None;
            let mut complete_ac_model: Option<Dac<R>>;
            let mut complete_ac_nonmodel: Option<Dac<R>>;
            let mut cnt_idem = 0;
            let mut last_val = 0.0;
            let mut last_discrepancy= 0;
            let mut lbs = vec![];
            let mut ubs = vec![];
            let mut lb_size = vec![];
            let mut ub_size = vec![];
            let mut epsilon = vec![];
            let mut time = vec![];
            loop {
                let solution = self.do_discrepancy_iteration(discrepancy);
                if (self.parameters.start.elapsed().as_secs() < self.parameters.timeout || complete_sol.as_ref().is_none()) && cnt_idem < 20 {
                    last_discrepancy = discrepancy;
                    complete_ac_model = Some(self.build_ac(solution.epsilon(), &forced_by_propagation, true, solution.cache_index(), last_discrepancy));
                    complete_ac_nonmodel = Some(self.build_ac(solution.epsilon(), &forced_by_propagation, false, solution.cache_index(), last_discrepancy));
                    lbs.push(solution.bounds().0);
                    ubs.push(solution.bounds().1);
                    lb_size.push(complete_ac_model.as_ref().unwrap().number_nodes());
                    ub_size.push(complete_ac_nonmodel.as_ref().unwrap().number_nodes());
                    epsilon.push(solution.epsilon());
                    time.push(start.elapsed().as_secs());
                    if solution.bounds().0 == last_val {
                        cnt_idem += 1;
                    }
                    else {
                        cnt_idem = 0;
                        last_val = solution.bounds().0;
                    }
                    complete_sol = Some(solution);
                }
                if self.parameters.start.elapsed().as_secs() >= self.parameters.timeout || complete_sol.as_ref().unwrap().has_converged(self.parameters.epsilon) || cnt_idem>=20 {//|| discrepancy == 2 {
                    let (lb_discr, ub_discr, eps) = if lds_opti {self.choose_discrepancy(&lbs, &ubs, &lb_size, &ub_size)} else {(last_discrepancy, last_discrepancy, complete_sol.as_ref().unwrap().epsilon())};
                    if lds_opti {
                        println!("lbs: {:?}", lbs);
                        println!("ubs: {:?}", ubs);
                        println!("lb_size: {:?}", lb_size);
                        println!("ub_size: {:?}", ub_size);
                        println!("epsilon: {:?}", epsilon);
                        println!("chosen lb: {}", lb_discr);
                        println!("chosen ub: {}", ub_discr);
                        println!("chosen lb pred: {}", lbs[lb_discr]);
                        println!("chosen ub pred: {}", ubs[ub_discr]);
                        println!("chosen epsilon: {}", eps);
                        println!("chosen lb size: {}", lb_size[lb_discr]);
                        println!("chosen ub size: {}", ub_size[ub_discr]);
                        println!("total size: {}", lb_size[lb_discr]+ub_size[ub_discr]);
                        println!("other size: {}", lb_size[lbs.len()-1]+ub_size[lbs.len()-1]);
                        println!("time: {:?}", time);
                    }
                    let mut complete_ac_model: Dac<R> = self.build_ac(eps, &forced_by_propagation, true, complete_sol.as_ref().unwrap().cache_index(), lb_discr);
                    let mut complete_ac_nonmodel: Dac<R> = self.build_ac(eps, &forced_by_propagation, false, complete_sol.as_ref().unwrap().cache_index(), ub_discr);
                    complete_ac_model.set_compile_time(start.elapsed().as_secs());
                    complete_ac_nonmodel.set_compile_time(start.elapsed().as_secs());
                    complete_ac_nonmodel.evaluate();
                    if !lds_opti{println!("solution {:?} model out {}, nonmodel out 1-{} = {}", complete_sol.as_ref().unwrap().bounds(), complete_ac_model.evaluate().to_f64(), complete_ac_nonmodel.evaluate().to_f64(), 1.0-complete_ac_nonmodel.evaluate().to_f64());}
                    return (complete_ac_model, complete_ac_nonmodel);
                }
                discrepancy += 1;
            }
        }
    }

    pub fn build_ac<R: SemiRing>(&mut self, epsilon: f64, forced_by_propagation:&(Vec<(usize, (usize, f64))>, Vec<(usize, Vec<(usize, f64)>)>, Vec<(usize, Vec<(usize, f64)>)>, Vec<(usize, Vec<(usize, f64)>)>), 
                                sat_compile:bool, cache_index:Option<usize>, discrepancy_limit:usize) -> Dac<R> {
        let prnt = false;// !sat_compile;
        let mut dac = Dac::new(epsilon);
        // Adds the distributions in the circuit
        /* for distribution in self.problem.distributions_iter() {
            for v in self.problem[distribution].iter_variables() {
                let _ = dac.distribution_value_node(&self.problem, distribution, v);
            }
        } */

        let mut has_node_search = false;
        /* let mut init_k = self.component_extractor[ComponentIndex(0)].get_cache_key();
        if let Some(s_k) = self.cache_keys.get(0) {
            if s_k.repr().contains(init_k.repr()) {
                init_k = s_k.clone();
            }
        } 
        let mut root_number_children = if self.cache.contains_key(&init_k) { has_node_search = true; 1 } else { 0 };*/
        let mut root_number_children = if let Some(_) = cache_index { has_node_search = true; 1 } else { 0 };
        root_number_children += forced_by_propagation.0.len() + forced_by_propagation.1.len();
        if prnt {println!("\nroot forced by propagation {:?}", forced_by_propagation);
        println!("\nroot number children {}", root_number_children);
        println!("has node search {}", has_node_search);}

        // The super root is used in case we have some excluded interpretations that will be added to the classical dac found from the cache exploration.
        // In such cases, the root variable below will be a child of the super root including the sum of the excluded interpretations
        let mut super_root = None;

        let excluded_interpretations = self.compute_excluded_interpretations_at_preprocessing(&forced_by_propagation.2, &forced_by_propagation.3);
        if prnt {println!("excluded interpretations {:?}", excluded_interpretations);}

        if !sat_compile && forced_by_propagation.2.len() > 0 && excluded_interpretations.len() > 0 {
            let super_nb_children = if has_node_search { 2 } else { 1 };
            if prnt {println!("super nb children {}", super_nb_children);}
            let super_sum_node = dac.sum_node(super_nb_children);
            // Adding a sum node to sum all the excluded interpretations
            let nb_excl_interpretations = excluded_interpretations.len();
            let sum_node_interpret = dac.sum_node(nb_excl_interpretations);
            let mut interpret_child_id = sum_node_interpret.input_start();    

            for interpret in excluded_interpretations {            
                let interpetation_prod = dac.prod_node(interpret.len());
                for (i,(d, values)) in interpret.iter().enumerate() {
                    let sum_distri_vals = dac.sum_node(values.len());
                    for (j, (v, w)) in values.iter().copied().enumerate() {
                        let distribution_val = dac.old_distribution_old_value_node(*d, v, w);
                        dac.add_input(sum_distri_vals.input_start() + j, distribution_val);
                    }
                    let id = dac.add_node(sum_distri_vals);
                    dac.add_input(interpetation_prod.input_start() + i, id);
                }
                let interpretation_prod_id = dac.add_node(interpetation_prod);
                dac.add_input(interpret_child_id, interpretation_prod_id);
                interpret_child_id += 1;
            }
            let excluded_preproc_id = dac.add_node(sum_node_interpret);
            dac.add_input(super_sum_node.input_start(), excluded_preproc_id);

            if !has_node_search {
                if prnt {println!("no node search");}
                let root = dac.add_node(super_sum_node);
                dac.set_root(root);
                return dac;
            }
            super_root = Some(super_sum_node);
        }

        let root = dac.prod_node(root_number_children);
        let mut child_id = root.input_start();
        if prnt {println!("has node search {}", has_node_search);}
        // Forced variables from the propagation
        for (d, (variable, w)) in forced_by_propagation.0.iter().copied() {
            let distribution_node = dac.old_distribution_old_value_node(d, variable, w);
            dac.add_input(child_id, distribution_node);
            child_id += 1;
        }
        // Unconstrained distribution variables
        for (d, values) in forced_by_propagation.1.iter() {
            let sum_node = dac.sum_node(values.len());
            for (i, (v, w)) in values.iter().copied().enumerate() {
                let distribution_node = dac.old_distribution_old_value_node(*d, v, w);
                dac.add_input(sum_node.input_start() + i, distribution_node);
            }
            let sum_id = dac.add_node(sum_node);
            dac.add_input(child_id, sum_id);
            child_id += 1;
        }

        let mut map: FxHashMap<usize, NodeIndex> = FxHashMap::default();
        let mut dist_map: FxHashMap<String, NodeIndex> = FxHashMap::default();
        if has_node_search {
            let node_search = self.explore_cache(&mut dac, cache_index.unwrap(), &mut map, &mut dist_map, sat_compile, discrepancy_limit);
            dac.add_input(child_id, node_search);
        }
        if let Some(super_root) = super_root {
            let root_id = dac.add_node(root);
            dac.add_input(super_root.input_start()+1, root_id);
            let super_id = dac.add_node(super_root);
            dac.set_root(super_id);
            return dac;
        }
        let root = dac.add_node(root);
        dac.set_root(root);
        dac
    }

    pub fn explore_cache<R: SemiRing>(&self, dac: &mut Dac<R>, cache_key_index: usize, c: &mut FxHashMap<usize, NodeIndex>, dist_c: &mut FxHashMap<String, NodeIndex>, 
                                    sat_compile: bool, discrepancy_limit:usize) -> NodeIndex {
        let prnt = false;// !sat_compile;// && cache_key_index == 0;
        if prnt {println!("\nexplore cache key index {}", cache_key_index);}
        if let Some(child_i) = c.get(&cache_key_index) {
            return *child_i;
        }
        let current = self.cache.get(&self.cache_keys[cache_key_index]).unwrap();

        if current.init_discrepancy >= discrepancy_limit {
            let p_node = dac.prod_node(0);
            let p_id = dac.add_node(p_node);
            return p_id;
        }
        if current.bounds.0.clone() + FLOAT_CMP_THRESHOLD >= current.max_probability && !sat_compile {
            let p_node = dac.prod_node(0);
            let p_id = dac.add_node(p_node);
            return p_id;
        }
        if sat_compile && current.number_children(sat_compile, discrepancy_limit) == 0 {
            // For partial compilation, we can have entry without any sat children -> Zero
            let p_node = dac.prod_node(0);
            let p_id = dac.add_node(p_node);
            return p_id;
        }
        if prnt {println!("current bounds {:?}", current.bounds);
        println!("current max prob {}", current.max_probability);}

        // Sum node with the different values the distribution can take
        let sum_node_nb_child = current.number_children(sat_compile, discrepancy_limit);
        if sum_node_nb_child == 0 && !sat_compile {
            //println!("no children");
            // For partial compilation, we can have entry without any children, which is a one when nonmodel compilation
            let p_node = dac.prod_node(0);
            let p_id = dac.add_node(p_node);
            return p_id;
        }
        let sum_node = dac.sum_node(sum_node_nb_child);

        if prnt {println!("sum node child size {}, sat_compile {}", sum_node_nb_child, sat_compile);
        println!("children variables {:?}", current.children_variables(true));
        println!("unsat children {:?}", current.children_variables(false));}

        let mut sum_node_child = 0;
        // Iterate on the variables the distribution with the associated cache key for the sat children
        for variable in current.children_variables(true){//[..sum_node_nb_child].iter().copied() {
            if prnt {println!("SAT PART");
            println!("var {}, discrepancy {}", variable.0, *current.child_discrepancy.get(&variable).unwrap_or(&0));}
            if *current.child_discrepancy.get(&variable).unwrap_or(&0) >= discrepancy_limit {continue;}
            if prnt{ println!("explore variable {}, distribution old i {}, var in d {}", variable.0, self.problem[current.distribution.unwrap()].old_index(), self.problem[variable].index_in_distribution().unwrap());}

            let number_children = if sat_compile {
                    current.child_keys(variable, true).len() + current.forced_choices(variable, true).len() + current.unconstrained_distribution_variables_of(variable, true).len()
                } else {
                    // Only link to a difference node instead of multiplying the independent components
                    let nb_keys = if current.child_keys(variable, true).len()>=1 {1} else {0};
                    nb_keys + current.forced_choices(variable, true).len() + current.unconstrained_distribution_variables_of(variable, true).len()
                };
            let number_sub_keys = current.child_keys(variable, true).len();
            if prnt {println!("number children {}", number_children);
            println!("nb child key {}, nb forced {}, nb unconstrained {}", current.child_keys(variable, true).len(), current.forced_choices(variable, true).len(), current.unconstrained_distribution_variables_of(variable, true).len());
            println!("forced choices {:?}", current.forced_choices(variable, true));
            println!("TODO stopping criterion total distributions of {:?}, ", current.total_distributions_of(variable, true));}
            if number_children == 0 && (sat_compile || current.total_distributions_of(variable, true).len() == 0) {
                if prnt {println!("we stop here");}
                continue;
            }
            let nb_sum_node = if !sat_compile && (current.total_distributions_of(variable, true).len()!=0 && number_sub_keys!=0) {2} else {1};
            // For the variable, we sum the excluded interpretations and the sat exploration
            let sum_node_variable = dac.sum_node(nb_sum_node);
            let mut sum_node_variable_child = 0;

            if prnt {println!("total distributions of variable {:?}", current.total_distributions_of(variable, true));
            println!("total remaining to remove of variable {:?}", current.total_remaining_to_remove_of(variable, true));
            println!("nb sum node {}", nb_sum_node);}

            if !sat_compile && current.total_distributions_of(variable, true).len()!=0 {
                // Multipling the excluded interpretations by the value that is branched on
                let prod_node_excl = dac.prod_node(2);
                let d_node = dac.distribution_value_node(&self.problem, current.distribution.unwrap(), variable);
                dac.add_input(prod_node_excl.input_start(), d_node);
                // Excluded models using before and after states, could be replaced by excluded interpretations
                let sum_node_excl = dac.sum_node(2);
                if prnt {println!("total distributions of variable {:?}", current.total_distributions_of(variable, true));
                println!("total remaining to remove of variable {:?}", current.total_remaining_to_remove_of(variable, true));}
                let before_node = self.one_interpretation_node(dac, current.total_distributions_of(variable, true), dist_c);
                dac.add_input(sum_node_excl.input_start(), before_node);
                let after_node;
                let mut to_remove = current.total_remaining_to_remove_of(variable, true).clone();
                for (d, v) in current.forced_choices(variable, true).iter().copied() {
                    if d != current.distribution.unwrap() {
                        to_remove.push((d, vec![v]));
                    }
                }
                if prnt {println!("to remove {:?}", to_remove);}
                if to_remove.len() == 0 {
                    let sum_node = dac.sum_node(0);
                    after_node = dac.add_node(sum_node);
                }
                else {
                    after_node = self.one_interpretation_node(dac, &to_remove, dist_c);
                }
                let negation_node = dac.opposite_node(1);
                dac.add_input(negation_node.input_start(), after_node);
                let negation_node_id = dac.add_node(negation_node);
                if prnt {println!("negation node id of exclusion {:?}", negation_node_id);}
                dac.add_input(sum_node_excl.input_start()+1, negation_node_id);
                let sum_node_excl_id = dac.add_node(sum_node_excl);
                dac.add_input(prod_node_excl.input_start()+1, sum_node_excl_id);
                let prod_node_excl_id = dac.add_node(prod_node_excl);
                dac.add_input(sum_node_variable.input_start() + sum_node_variable_child, prod_node_excl_id);
                sum_node_variable_child += 1;

            }

            // Focusing on the sat exploration
            // If we are compiling the non-models and the remaining part is fully sat, we do not explore the sat part
            if prnt {println!("current bounds {:?}", current.bounds);
            println!("current max prob {}", current.max_probability);}
            if !sat_compile && (number_sub_keys == 0 || current.bounds.0 >= current.max_probability)  {//number_children == 0 || (!sat_compile && variable.0 == 6) {
                if prnt {println!("as non-model compilation, no exploration of sat part because fully sat or no children, sum node variable child {} total size {}", sum_node_variable_child, nb_sum_node);}
                let sum_node_variable_id = dac.add_node(sum_node_variable);
                dac.add_input(sum_node.input_start() + sum_node_child, sum_node_variable_id);
                sum_node_child += 1;
                continue;
            }

            // Product node to multiply the propagated variables
            let prod_node = dac.prod_node(number_children);

            let mut child_id = prod_node.input_start();
            // Adding to the new product node all the propagated variables, including the distribution value we branch on
            for (d, v) in current.forced_choices(variable, true).iter().copied() {
                let distribution_prop = dac.distribution_value_node(&self.problem, d, v);
                dac.add_input(child_id, distribution_prop);
                child_id += 1;

            }

            // Adding to the new product node sum nodes for all the unconstrained distribution not summing to 1
            for (d, values) in current.unconstrained_distribution_variables_of(variable, true).iter() {
                let sum_node_unconstrained = dac.sum_node(values.len());
                for (i, v) in values.iter().copied().enumerate() {
                    let distribution_unconstrained = dac.distribution_value_node( &self.problem, *d, v);
                    dac.add_input(sum_node_unconstrained.input_start() + i, distribution_unconstrained);
                }
                let id = dac.add_node(sum_node_unconstrained);
                dac.add_input(child_id, id);
                child_id += 1;
            }

            // Recursively build the DAC for each sub-component in case of a sat compilation
            if sat_compile {
                for cache_key in current.child_keys(variable, true) {
                    let id = self.explore_cache(dac, cache_key, c, dist_c, sat_compile, discrepancy_limit);
                    dac.add_input(child_id, id);
                    child_id += 1;
                }
            }
            else {
                // Create a difference node between the before state and the product of the independent components contributions
                let diff_node = dac.sum_node(2);
                let before_node = self.one_interpretation_node(dac, current.total_remaining_to_remove_of(variable, true), dist_c);
                if prnt {println!("before node id {:?}, total to remove {:?}", before_node, current.total_remaining_to_remove_of(variable, true));}
                dac.add_input(diff_node.input_start(), before_node);
                let oppostion_node = dac.opposite_node(1);
                let prod_components = dac.prod_node(current.child_keys(variable, true).len());
                let mut child_prod_id = prod_components.input_start();
                for cache_key in current.child_keys(variable, true) {
                    // Difference between the before of the subcomponent and the classical exploration
                    let sum_component = dac.sum_node(2);
                    let component = self.cache.get(&self.cache_keys[cache_key]).unwrap();
                    // The before of all the subcomponents is the remaining values of the considered cache key
                    let before_component_id = self.one_interpretation_node(dac, &component.remaining_dist_values, dist_c);
                    if prnt {println!("before component id {:?}", before_component_id);}
                    dac.add_input(sum_component.input_start(), before_component_id);
                    let oppositer_after = dac.opposite_node(1);
                    let id = self.explore_cache(dac, cache_key, c, dist_c, sat_compile, discrepancy_limit);
                    if prnt {println!("explore cache minus id {:?}", id);}
                    dac.add_input(oppositer_after.input_start(), id);
                    let oppositer_after_id = dac.add_node(oppositer_after);
                    if prnt {println!("oppositer after id {:?}", oppositer_after_id);}
                    dac.add_input(sum_component.input_start()+1, oppositer_after_id);
                    let sum_component_id = dac.add_node(sum_component);
                    dac.add_input(child_prod_id, sum_component_id);
                    child_prod_id += 1;
                }
                let prod_components_id = dac.add_node(prod_components);
                dac.add_input(oppostion_node.input_start(), prod_components_id);
                let oppostion_node_id = dac.add_node(oppostion_node);
                if prnt {println!("global opposition component node id {:?}", oppostion_node_id);}
                dac.add_input(diff_node.input_start()+1, oppostion_node_id);
                let diff_node_id = dac.add_node(diff_node);
                dac.add_input(child_id, diff_node_id);
                //child_id += 1;
            }

            // Adding the product node of classical exploration to the variable sum node and the variable sum to the global sum node
            let prod_id = dac.add_node(prod_node);
            dac.add_input(sum_node_variable.input_start()+sum_node_variable_child, prod_id);
            let sum_node_variable_id = dac.add_node(sum_node_variable);
            dac.add_input(sum_node.input_start() + sum_node_child, sum_node_variable_id);
            sum_node_child += 1;
        }
        // Iterate on the variables the distribution with the associated cache key for the unsat children in case of an unsat compilation
        if !sat_compile {
            if prnt {if current.children_variables(false).len() != 0 {println!("UNSAT PART");
                    println!("unsat children {:?}", current.children_variables(false));}}
            for variable in current.children_variables(false) {
                if prnt {println!("explore variable {}", variable.0);}
                let nb_sum = if current.child_keys(variable, false).len() == 0 {1} else {2};
                let sum_unsat = dac.sum_node(nb_sum);
                let mut sum_unsat_child = 0;
                if nb_sum != 1 {
                    let removed_prod = dac.prod_node(2);
                    let d_node = dac.distribution_value_node(&self.problem, current.distribution.unwrap(), variable);
                    dac.add_input(removed_prod.input_start(), d_node);
                    let removed_sum = dac.sum_node(2);
                    let before_node = self.one_interpretation_node(dac, current.total_distributions_of(variable, false), dist_c);
                    dac.add_input(removed_sum.input_start(), before_node);
                    let negation_node = dac.opposite_node(1);
                    let after_node = self.one_interpretation_node(dac, current.total_remaining_to_remove_of(variable, false), dist_c);
                    dac.add_input(negation_node.input_start(), after_node);
                    let negation_node_id = dac.add_node(negation_node);
                    dac.add_input(removed_sum.input_start()+1, negation_node_id);
                    let removed_sum_id = dac.add_node(removed_sum);
                    dac.add_input(removed_prod.input_start()+1, removed_sum_id);
                    let removed_prod_id = dac.add_node(removed_prod);
                    dac.add_input(sum_unsat.input_start(), removed_prod_id);
                    sum_unsat_child += 1;
                }
                let number_children = current.variable_unsat_number_children(variable, false);
                if number_children == 0 {
                    continue;
                }
                let prod_node = dac.prod_node(number_children);
                if prnt {
                    println!("forced choices {:?}", current.forced_choices(variable, false));
                    println!("unsat children {:?}", current.child_keys(variable, false));
                    println!("excluded distributions {:?}", current.excluded_distribution_variables_of(variable, false));
                    println!("unc distributions {:?}", current.unconstrained_distribution_variables_of(variable, false));
                    println!("total distributions {:?}", current.total_distributions_of(variable, false));
                    println!("total remaining to remove {:?}", current.total_remaining_to_remove_of(variable, false));
                }

                let mut child_id = prod_node.input_start();
                // Adding to the new product node all the propagated variables, including the distribution value we branch on
                for (d, v) in current.forced_choices(variable, false).iter().copied() {
                    if prnt {println!("forced distr d {} v {}", d.0, v.0);}
                    let distribution_prop = dac.distribution_value_node(&self.problem, d, v);
                    dac.add_input(child_id, distribution_prop);
                    child_id += 1;

                }

                for (d, values) in current.unconstrained_distribution_variables_of(variable, false).iter() {
                    let sum_node_unconstrained = dac.sum_node(values.len());
                    for (i, v) in values.iter().copied().enumerate() {
                        let distribution_unconstrained = dac.distribution_value_node( &self.problem, *d, v);
                        dac.add_input(sum_node_unconstrained.input_start() + i, distribution_unconstrained);
                    }
                    let id = dac.add_node(sum_node_unconstrained);
                    dac.add_input(child_id, id);
                    child_id += 1;
                }

                // Adding to the new product node sum nodes for all the excluded distribution if unsat mode
                for (d, values) in current.excluded_distribution_variables_of(variable, false).iter() {
                    let d_key = String::from("d") + self.problem[*d].old_index().to_string().as_str() + values.iter().map(|v| String::from("v")+self.problem[*v].old_index().to_string().as_str()).collect::<String>().as_str();
                    if prnt {println!("unsat key {}", d_key);}
                    if let Some(node_id) = dist_c.get(&d_key) {
                        dac.add_input(child_id, *node_id);
                        child_id += 1;
                        continue;
                    }
                    let sum_node_excluded = dac.sum_node(values.len());
                    for (i, v) in values.iter().copied().enumerate() {
                        if prnt {println!("excluded distr {} value {}", d.0, v.0);}
                        let distribution_excluded = dac.distribution_value_node(&self.problem, *d, v);
                        dac.add_input(sum_node_excluded.input_start() + i, distribution_excluded);
                    }
                    let id = dac.add_node(sum_node_excluded);
                    dac.add_input(child_id, id);
                    child_id += 1;
                    dist_c.insert(d_key, id);
                }
                // Recursively build the DAC for each sub-component
                if prnt {println!("unsat children {:?}", current.child_keys(variable, sat_compile));}
                for cache_key in current.child_keys(variable, sat_compile) {
                    let id = self.explore_cache(dac, cache_key, c, dist_c, false, discrepancy_limit);
                    dac.add_input(child_id, id);
                    child_id += 1;
                }
                let id = dac.add_node(prod_node);
                if prnt {println!("UNSAT child id {:?}", id);}
                dac.add_input(sum_unsat.input_start() + sum_unsat_child, id);
                let unsat_id = dac.add_node(sum_unsat);
                dac.add_input(sum_node.input_start() + sum_node_child, unsat_id);
                sum_node_child += 1;
            }
        }
        let sum_index = dac.add_node(sum_node);
        c.insert(cache_key_index, sum_index);
        sum_index
    }

    fn one_interpretation_node<R: SemiRing>(&self, dac: &mut Dac<R>, distribution_values: &Vec<(DistributionIndex, Vec<VariableIndex>)>, dist_c: &mut FxHashMap<String, NodeIndex>) -> NodeIndex {
        let i_key = distribution_values.iter().map(|(d, values)| String::from("d")+self.problem[*d].old_index().to_string().as_str()+values.iter().map(|v| String::from("v")+self.problem[*v].old_index().to_string().as_str()).collect::<String>().as_str()).collect::<String>();
        if let Some(node_id) = dist_c.get(&i_key) {
            return *node_id;
        }
        let prod_node = dac.prod_node(distribution_values.len());
        let mut child_id = prod_node.input_start();
        for (d, values) in distribution_values.iter() {
            let d_key = String::from("d") + self.problem[*d].old_index().to_string().as_str() + values.iter().map(|v| String::from("v")+self.problem[*v].old_index().to_string().as_str()).collect::<String>().as_str();
            if let Some(node_id) = dist_c.get(&d_key) {
                dac.add_input(child_id, *node_id);
                child_id += 1;
                continue;
            }
            let sum_node = dac.sum_node(values.len());
            for (i, v) in values.iter().copied().enumerate() {
                let distribution_val = dac.distribution_value_node(&self.problem, *d, v);
                dac.add_input(sum_node.input_start() + i, distribution_val);
            }
            let sum_id = dac.add_node(sum_node);
            dac.add_input(child_id, sum_id);
            child_id += 1;
            dist_c.insert(d_key, sum_id);
        }
        let prod_id = dac.add_node(prod_node);
        dist_c.insert(i_key, prod_id);
        prod_id
    }

    fn compute_excluded_interpretations_at_preprocessing(&mut self, excluded_distributions: &Vec<(usize, Vec<(usize, f64)>)>, distributions_remaining_vals: &Vec<(usize, Vec<(usize, f64)>)>) -> Vec<Vec<(usize, Vec<(usize, f64)>)>> {
        let mut excluded_interpretations = vec![];
        let excl_d = excluded_distributions.iter().map(|(d, _)| *d).collect::<Vec<usize>>();
        let mut seen_d = vec![];
        for (d, values) in excluded_distributions.iter() {
            let mut interpretations = vec![];
            interpretations.push((*d, values.clone()));
            for (r_d, r_values) in distributions_remaining_vals.iter() {
                if !values.is_empty() && *r_d != *d && (!excl_d.contains(r_d) || seen_d.contains(r_d)) {
                    interpretations.push((*r_d, r_values.clone()));
                }
            }
            excluded_interpretations.push(interpretations);
            seen_d.push(*d);
        }
        excluded_interpretations
    }

    /// Returns the choices (i.e., assignments to the distributions) made during the propagation as
    /// well as the distributions that are not constrained anymore.
    /// A choice for a distribution is a pair (d, i) = (DistributionIndex, usize) that indicates that
    /// the i-th value of disitribution d is true.
    /// An unconstrained distribution is a pair (d, v) = (DistributionIndex, Vec<usize>) that
    /// indicates that distribution d does not appear in any clauses and its values in v are not
    /// set yet.
    fn forced_from_propagation(&mut self, current_distribution: DistributionIndex, is_sat:bool) -> (Vec<DistributionChoice>, Vec<UnconstrainedDistribution>, Vec<UnconstrainedDistribution>) {
        let mut forced_distribution_variables: Vec<DistributionChoice> = vec![];
        let mut unconstrained_distribution_variables: Vec<UnconstrainedDistribution> = vec![];
        let mut excluded_distribution_variables: Vec<UnconstrainedDistribution> = vec![];
        
        if self.propagator.has_assignments(&self.state) || self.propagator.has_unconstrained_distribution() {
            // First, we look at the assignments
            for literal in self.propagator.assignments_iter(&self.state) {
                let variable = literal.to_variable();
                // Only take probabilistic variables set to true
                if self.problem[variable].is_probabilitic() && literal.is_positive() && self.problem[variable].weight().unwrap() != 1.0 && is_sat {
                    let distribution = self.problem[variable].distribution().unwrap();
                    // This represent which "probability index" is send to the node
                    forced_distribution_variables.push((distribution, variable));
                }
                else if self.problem[variable].is_probabilitic() && !literal.is_positive() && self.problem[variable].distribution().unwrap() != current_distribution {
                    let distribution = self.problem[variable].distribution().unwrap();
                    let position = excluded_distribution_variables.iter().position(|(d, _)| *d == distribution);
                    if let Some(i) = position {
                        excluded_distribution_variables[i].1.push(variable);
                    }
                    else {
                        excluded_distribution_variables.push((distribution, vec![variable]));
                    }
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
        (forced_distribution_variables, unconstrained_distribution_variables, excluded_distribution_variables)
    }

    pub fn choose_discrepancy(&self, lbs: &Vec<f64>, ubs: &Vec<f64>, lbs_size: &Vec<usize>, ubs_size: &Vec<usize>) -> (usize, usize, f64) {
        let mut lb_disc = 0;
        let mut ub_disc = 0;
        while lbs_size[lb_disc] < 1000 && lb_disc < lbs.len()-1{
            lb_disc += 1;
        }
        while ubs_size[ub_disc] < 1000 && ub_disc < ubs.len()-1{
            ub_disc += 1;
        }
        let lb_gain = (0..lbs.len()-1).map(|i| 1.0-(lbs[i]/lbs[i+1])).collect::<Vec<f64>>();
        let ub_gain = (0..ubs.len()-1).map(|i| 1.0-(ubs[i+1]/ubs[i])).collect::<Vec<f64>>();
        let lb_size_gain = (0..lbs_size.len()-1).map(|i| 1.0-(lbs_size[i] as f64/lbs_size[i+1] as f64)).collect::<Vec<f64>>();
        let ub_size_gain = (0..ubs_size.len()-1).map(|i| 1.0-(ubs_size[i] as f64/ubs_size[i+1] as f64)).collect::<Vec<f64>>();
        println!("lb gain: {:?}", lb_gain);
        println!("ub gain: {:?}", ub_gain);
        println!("lb size gain: {:?}", lb_size_gain);
        println!("ub size gain: {:?}", ub_size_gain);
        let mut tresh = if lbs_size[lb_disc] < 2500 {0.02} else {0.1};
        while lb_disc < lbs.len()-1 && lb_gain[lb_disc] > tresh {
            lb_disc += 1;
            tresh = if lbs_size[lb_disc] < 2500 {0.02} else {0.1};
        }
        tresh = if ubs_size[ub_disc] < 2500 {0.02} else {0.1};
        while ub_disc < ubs.len()-1 && ub_gain[ub_disc] > tresh {
            ub_disc += 1;
            tresh = if ubs_size[ub_disc] < 2500 {0.02} else {0.1};
        }
        let eps = (ubs[ub_disc]/(lbs[lb_disc])).sqrt()-1.0;
        (lb_disc, ub_disc, eps)
    }
}

/// Parameters for the solving
pub struct SolverParameters {
    /// Memory limit for the solving, in megabytes. When reached, the cache is cleared. Note that
    /// this parameter should not be used for compilation.
    memory_limit: u64,
    /// Approximation factor
    epsilon: f64,
    /// Time limit for the search
    timeout: u64,
    /// Time at which the solving started
    start: Instant,
}

impl SolverParameters {
    pub fn new(memory_limit: u64, epsilon: f64, timeout: u64) -> Self {
        Self {
            memory_limit,
            epsilon,
            timeout,
            start: Instant::now(),
        }
    }
}

/// An entry in the cache for the search. It contains the bounds computed when the sub-problem was
/// explored as well as various informations used by the solvers.
#[derive(Clone)]
pub struct CacheEntry {
    /// The current bounds on the sub-problem
    bounds: Bounds,
    /// Maximum discrepancy used for that node
    discrepancy: usize,
    /// The distribution on which to branch in this problem
    distribution: Option<DistributionIndex>,
    children: FxHashMap<VariableIndex, CacheChildren>,
    cache_key_index: usize,
    unsat_children: FxHashMap<VariableIndex, CacheChildren>,
    max_probability: f64,
    remaining_dist_values: Vec<(DistributionIndex, Vec<VariableIndex>)>,
    init_discrepancy: usize,
    child_discrepancy: FxHashMap<VariableIndex,usize>,
}

impl CacheEntry {

    /// Returns a new cache entry
    pub fn new(bounds: Bounds, discrepancy: usize, distribution: Option<DistributionIndex>, children: FxHashMap<VariableIndex, CacheChildren>, 
                cache_key_index: usize, unsat_children: FxHashMap<VariableIndex,CacheChildren>, max_probability: f64, remaining_d_v: Vec<(DistributionIndex, Vec<VariableIndex>)>, 
                init_discrepancy: usize, child_discrepancy: FxHashMap<VariableIndex,usize>) -> Self {
        Self {
            bounds,
            discrepancy,
            distribution,
            children,
            cache_key_index,
            unsat_children,
            max_probability,
            remaining_dist_values: remaining_d_v,
            init_discrepancy,
            child_discrepancy,
        }
    }

    pub fn forced_choices(&self, variable: VariableIndex, sat_compile: bool) -> Vec<DistributionChoice> {
        if sat_compile {
            return self.children.get(&variable).unwrap().forced_choices.clone();
        }
        else {
            return self.unsat_children.get(&variable).unwrap().forced_choices.clone();
        }
    }

    pub fn unconstrained_distribution_variables_of(&self, variable: VariableIndex, sat_compile: bool) -> &Vec<UnconstrainedDistribution> {
        if sat_compile {
            return &self.children.get(&variable).unwrap().unconstrained_distributions;
        }
        else {
            return &self.unsat_children.get(&variable).unwrap().unconstrained_distributions;
        }
    }

    pub fn excluded_distribution_variables_of(&self, variable: VariableIndex, sat_compile: bool) -> &Vec<UnconstrainedDistribution> {
        if sat_compile {
            return &self.children.get(&variable).unwrap().excluded_distributions;
        }
        else {
            return &self.unsat_children.get(&variable).unwrap().excluded_distributions;
        }
    }

    pub fn total_distributions_of(&self, variable: VariableIndex, sat_compile: bool) -> &Vec<UnconstrainedDistribution> {
        if sat_compile {
            return &self.children.get(&variable).unwrap().total_distributions;
        }
        else {
            return &self.unsat_children.get(&variable).unwrap().total_distributions;
        }
    }

    pub fn total_remaining_to_remove_of(&self, variable: VariableIndex, sat_compile: bool) -> &Vec<UnconstrainedDistribution> {
        if sat_compile {
            return &self.children.get(&variable).unwrap().total_remaining_to_remove;
        }
        else {
            return &self.unsat_children.get(&variable).unwrap().total_remaining_to_remove;
        }
    }

    pub fn number_children(&self, sat_compile: bool, discrepancy_limit:usize) -> usize {
        let mut cnt = 0;
        for k in self.children.keys(){
            if *self.child_discrepancy.get(k).unwrap_or(&0) < discrepancy_limit {
                cnt += 1;
            }
        }
        if sat_compile {
            //return self.children.len();
            return cnt;
        }
        else {
            //return self.children.len() + self.unsat_children.len();
            return cnt + self.unsat_children.len();
        }
    }

    pub fn clear_children(&mut self) {
        self.children.clear();
        self.unsat_children.clear();
    }

    pub fn variable_unsat_number_children(&self, variable: VariableIndex, sat_compile: bool) -> usize {
        let entry;
        if sat_compile {
            entry = self.unsat_children.get(&variable).unwrap();
            entry.children_keys.len() + entry.forced_choices.len() + entry.unconstrained_distributions.len()
        }
        else {
            entry = self.unsat_children.get(&variable).unwrap();
            entry.children_keys.len() + entry.forced_choices.len() + entry.unconstrained_distributions.len() + entry.excluded_distributions.len()
        }
    }

    pub fn children_variables(&self, sat_compile: bool) -> Vec<VariableIndex> {
        if sat_compile {
            return self.children.keys().copied().collect();
        }
        else {
            return self.unsat_children.keys().copied().collect();
        }
    }

    pub fn child_keys(&self, variable: VariableIndex, sat_compile: bool) -> Vec<usize> {
        if sat_compile {
            return self.children.get(&variable).unwrap().children_keys.clone();
        }
        else {
            return self.unsat_children.get(&variable).unwrap().children_keys.clone();
        }
    }

    fn are_bounds_tight(&self, maximum_probability: f64) -> bool {
        (self.bounds.0.clone() + &self.bounds.1 - maximum_probability).abs() <= FLOAT_CMP_THRESHOLD
    }
    
}

#[derive(Default, Clone)]
pub struct CacheChildren {
    children_keys: Vec<usize>,
    forced_choices: Vec<DistributionChoice>,
    unconstrained_distributions: Vec<UnconstrainedDistribution>,
    excluded_distributions: Vec<UnconstrainedDistribution>,
    total_distributions: Vec<UnconstrainedDistribution>,
    total_remaining_to_remove: Vec<UnconstrainedDistribution>,
}

impl CacheChildren {
    pub fn new(forced_choices: Vec<DistributionChoice>, unconstrained_distributions: Vec<UnconstrainedDistribution>, excluded_distributions: Vec<UnconstrainedDistribution>) -> Self {
        Self {
            children_keys: vec![],
            forced_choices,
            unconstrained_distributions,
            excluded_distributions,
            total_distributions: vec![],
            total_remaining_to_remove: vec![],
        }
    }

    pub fn add_key(&mut self, key: usize) {
        self.children_keys.push(key);
    }
}

struct SearchResult {
    bounds: (Float, Float),
    backtrack_level: isize,
    cache_index: usize,
}
