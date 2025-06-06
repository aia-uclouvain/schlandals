use rustc_hash::FxHashMap;
use search_trail::{SaveAndRestore, StateManager};

use crate::statistics::Statistics;
use crate::branching::BranchingDecision;
use crate::common::*;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::problem::{DistributionIndex, Problem, VariableIndex};
use crate::ac::ac::{NodeIndex, Dac};
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use crate::PEAK_ALLOC;
use crate::caching::CacheKey;
use malachite::rational::Rational;
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
pub struct Solver<const S: bool, const C: bool> {
    /// Implication problem of the (Horn) clauses in the input
    problem: Problem,
    /// Manages (save/restore) the states (e.g., reversible primitive types)
    state: StateManager,
    /// Extracts the connected components in the problem
    component_extractor: ComponentExtractor,
    /// Heuristics that decide on which distribution to branch next
    branching_heuristic: Box<dyn BranchingDecision>,
    /// Runs Boolean Unit Propagation and Schlandals' specific propagation at each decision node
    propagator: Propagator,
    cache: FxHashMap<CacheKey, CacheEntry>,
    /// Statistics gathered during the solving
    statistics: Statistics<S>,
    /// Product of the weight of the variables set to true during propagation
    preproc_in: Option<Rational>,
    /// Probability of removed interpretation during propagation
    preproc_out: Option<Rational>,
    /// Parameters of the solving
    parameters: SolverParameters,
    /// The caches present in the cache. Used during compilation to reconstruct the AC from the
    /// cache (follow the children of a node)
    cache_keys: Vec<CacheKey>,
    bound_approx: bool,
}

impl<const S: bool, const C: bool> Solver<S, C> {
    pub fn new(
        problem: Problem,
        state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: Box<dyn BranchingDecision>,
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
            bound_approx: false,
        }
    }

    /// Restores the state of the solver to the previous state
    fn restore(&mut self) {
        self.propagator.restore(&self.state);
        self.state.restore_state();
    }

    /// Solves the problem represented by this solver using a DPLL-search based method.
    pub fn search(&mut self, is_lds: bool) -> Solution {
        let max = self.problem.distributions_iter().map(|d| rational(self.problem[d].remaining(&self.state))).product::<Rational>();
        self.state.save_state();
        if let Some(sol) = self.preprocess(&max) {
            self.statistics.print();
            return sol;
        }
        self.restructure_after_preprocess();

        if self.problem.number_clauses() == 0 {
            let lb = self.preproc_in.clone().unwrap();
            let ub = max - self.preproc_out.clone().unwrap();
            self.statistics.print();
            return Solution::new(lb, ub, self.parameters.start.elapsed().as_secs());
        }
        if !is_lds {
            if self.parameters.epsilon > 0.0 {
                self.bound_approx = true;
            }
            let sol = self.do_discrepancy_iteration(usize::MAX, self.parameters.epsilon);
            self.statistics.peak_memory(PEAK_ALLOC.peak_usage_as_mb());
            self.statistics.lower_bound(sol.bounds().0);
            self.statistics.upper_bound(sol.bounds().1);
            self.statistics.print();
            sol
        } else {
            let mut discrepancy = 1;
            let mut complete_sol = None;
            loop {
                let solution = self.do_discrepancy_iteration(discrepancy, 0.0);
                if solution.epsilon() < 0.01 {
                    discrepancy = usize::MAX;
                } else {
                    discrepancy += 1;
                }
                if self.parameters.start.elapsed().as_secs() < self.parameters.timeout || complete_sol.as_ref().is_none() {
                    solution.print();
                    complete_sol = Some(solution);
                }
                if self.parameters.start.elapsed().as_secs() >= self.parameters.timeout || complete_sol.as_ref().unwrap().has_converged(self.parameters.epsilon) {
                    self.statistics.peak_memory(PEAK_ALLOC.peak_usage_as_mb());
                    self.statistics.print();
                    return complete_sol.unwrap()
                }
            }
        }
    }

    /// Preprocess the problem, if the problem is solved during the preprocess, return a solution.
    /// Returns None otherwise
    fn preprocess(&mut self, max: &Rational) -> Option<Solution> {
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
                rational(0.0),
                rational(0.0),
                self.parameters.start.elapsed().as_secs(),
            ));
        }
        self.preproc_in = Some(preproc.unwrap());
        let max_after_preproc= self.problem.distributions_iter().map(|d| {
            rational(self.problem[d].remaining(&self.state))
        }).product::<Rational>();
        self.preproc_out = Some(max - max_after_preproc);
        None
    }

    fn restructure_after_preprocess(&mut self) {
        self.problem.clear_after_preprocess(&mut self.state);
        let distribution_max = self.problem.distributions_iter().map(|d| {
            rational(self.problem[d].remaining(&self.state))
        }).collect::<Vec<Rational>>();
        self.state.restore_state();
        for (id, distribution) in self.problem.distributions_iter().enumerate() {
            self.problem[distribution].set_remaining(distribution_max[id].clone(), &mut self.state);
        }
        let max_probability = distribution_max.iter().product::<Rational>();
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

        for clause in self.problem.clauses_iter() {
            if self.problem[clause].iter().filter(|l| l.is_positive()).count() == 0 {
                self.problem[clause].set_head_f_reachable(&mut self.state);
            }
            let number_deterministic_in_body = self.problem[clause].iter().filter(|l| !l.is_positive() && !self.problem[l.to_variable()].is_probabilitic()).count();
            self.problem[clause].refresh_number_deterministic_in_body(number_deterministic_in_body, &mut self.state);
        }
    }

    pub fn do_discrepancy_iteration(&mut self, discrepancy: usize, eps: f64) -> Solution {
        let max = self.problem.distributions_iter().map(|d| self.problem[d].remaining(&self.state)).product::<f64>();
        let result = self.pwmc(ComponentIndex(0), 1, discrepancy, eps);
        let p_in = result.bounds.0.clone();
        let p_out = result.bounds.1.clone();
        let lb = p_in * self.preproc_in.clone().unwrap();
        let ub: Rational = rational(max) - (self.preproc_out.clone().unwrap() + p_out * self.preproc_in.clone().unwrap());
        //let ub: Rational = max - (self.preproc_out.clone().unwrap() + p_out * self.preproc_in.clone().unwrap());
        Solution::new(lb, ub, self.parameters.start.elapsed().as_secs())
    }

    fn pwmc(&mut self, component: ComponentIndex, level: isize, discrepancy: usize, eps: f64) -> SearchResult {
        if PEAK_ALLOC.current_usage_as_mb() as u64 >= self.parameters.memory_limit {
            self.cache.clear();
        }
        let cache_key = self.component_extractor[component].get_cache_key();
        self.statistics.cache_access();
        let mut cache_entry = self.cache.remove(&cache_key).unwrap_or_else(|| {
            self.statistics.cache_miss();
            let cache_key_index = self.cache_keys.len();
            if C {
                self.cache_keys.push(cache_key.clone());
            }
            CacheEntry::new((rational(0.0), rational(0.0)), 0, None, FxHashMap::default(), cache_key_index)
        });
        if cache_entry.distribution.is_none() {
            self.statistics.or_node();
            cache_entry.distribution = self.branching_heuristic.branch_on(&self.problem, &mut self.state, &self.component_extractor, component);
        }
        let mut complete = cache_entry.discrepancy < discrepancy;
        if cache_entry.discrepancy < discrepancy && !cache_entry.is_complete() {
            let mut new_p_in = rational(0.0);
            let mut new_p_out = rational(0.0);
            let distribution = cache_entry.distribution.unwrap();
            let unsat_factor = self.component_extractor.component_distribution_iter(component).filter(|d| *d != distribution).map(|d| {
                rational(self.problem[d].remaining(&self.state))
            }).product::<Rational>();
            let max_probability = self.component_extractor.component_distribution_iter(component).map(|d| {
                rational(self.problem[d].remaining(&self.state))
            }).product::<Rational>();
            let mut child_id = 0;
            for variable in self.problem[distribution].iter_variables() {
                if self.problem[variable].is_fixed(&self.state) {
                    continue;
                }
                if self.parameters.start.elapsed().as_secs() >= self.parameters.timeout || child_id == discrepancy {
                    complete = false;
                    break;
                }
                if self.bound_approx {
                    let ub = max_probability.clone() - new_p_out.clone();
                    let lb = new_p_in.clone();
                    if ub <= lb*rational((1.0 + eps)*(1.0 + eps)) {
                        complete = false;
                        break;
                    }
                }
                let v_weight = self.problem[variable].weight().unwrap();
                self.state.save_state();
                match self.propagator.propagate_variable(variable, true, &mut self.problem, &mut self.state, component, &mut self.component_extractor, level) {
                    Err(_) => {
                        self.statistics.unsat();
                        new_p_out += v_weight * unsat_factor.clone();
                    },
                    Ok(_) => {
                        let p = self.propagator.get_propagation_prob();
                        let removed = unsat_factor.clone() - self.component_extractor
                            .component_distribution_iter(component)
                            .filter(|d| *d != distribution)
                            .map(|d| rational(self.problem[d].remaining(&self.state)))
                            .product::<Rational>();
                        new_p_out += removed * v_weight;

                        // Decomposing into independent components
                        let mut prod_p_in = rational(1.0);
                        let mut prod_p_out = rational(1.0);
                        let prod_maximum_probability = self.component_extractor
                            .component_distribution_iter(component)
                            .filter(|d| self.problem[*d].is_constrained(&self.state))
                            .map(|d| rational(self.problem[d].remaining(&self.state)))
                            .product::<Rational>();
                        let (forced_distribution_var, unconstrained_distribution_var) = self.forced_from_propagation();
                        let mut child_entry = CacheChildren::new(forced_distribution_var, unconstrained_distribution_var);
                        self.state.save_state();
                        if self.component_extractor.detect_components(&mut self.problem, &mut self.state, component) {
                            self.statistics.decomposition(self.component_extractor.number_components(&self.state));
                            let number_components = self.component_extractor.number_components(&self.state);
                            let new_eps = eps.powf(1.0 / number_components as f64);
                            for sub_component in self.component_extractor.components_iter(&self.state) {
                                let sub_maximum_probability = self.component_extractor[sub_component].max_probability();
                                let sub_solution = self.pwmc(sub_component, level + 1, discrepancy - child_id, new_eps);
                                if !sub_solution.complete {
                                    complete = false;
                                }
                                prod_p_in *= &sub_solution.bounds.0;
                                prod_p_out *= sub_maximum_probability - &sub_solution.bounds.1;
                                if prod_p_in == 0.0 {
                                    break;
                                }
                                if C {
                                    child_entry.add_key(sub_solution.cache_index);
                                }
                            }
                        }
                        if C && prod_p_in > 0.0 {
                            cache_entry.children.insert(variable, child_entry);
                        }
                        prod_p_out = prod_maximum_probability - prod_p_out;
                        new_p_in += prod_p_in * &p;
                        new_p_out += prod_p_out * &p;
                        self.restore();
                    },
                }
                self.restore();
                child_id += 1;
            }
            cache_entry.discrepancy = discrepancy;
            cache_entry.bounds = (new_p_in, new_p_out);
        }
        let result = SearchResult {
            bounds: cache_entry.bounds.clone(),
            cache_index: cache_entry.cache_key_index,
            complete,
        };
        if complete {
            cache_entry.completed();
        }
        self.cache.insert(cache_key, cache_entry);
        result
    }
}

impl<const S: bool, const C: bool> Solver<S, C> {

    pub fn compile(&mut self, is_lds: bool) -> Dac {
        let start = Instant::now();
        let max = self.problem.distributions_iter().map(|d| rational(self.problem[d].remaining(&self.state))).product::<Rational>();
        self.state.save_state();
        let preproc_result = self.preprocess(&max);
        // Create the DAC and add elements from the preprocessing
        let forced_by_propagation = self.forced_from_propagation();
        self.restructure_after_preprocess();
        if preproc_result.is_some() {
            return Dac::default();
        }
        // Perform the actual search that will fill the cache
        if self.problem.number_clauses() == 0 {
            let mut ac = self.build_ac(0.0, &forced_by_propagation);
            ac.set_compile_time(start.elapsed().as_secs());
            return ac
        }
        if !is_lds {
            let sol = self.do_discrepancy_iteration(usize::MAX, self.parameters.epsilon);
            self.statistics.print();
            if sol.has_converged(0.0) && sol.bounds().0 < FLOAT_CMP_THRESHOLD {
                let mut ac = Dac::default();
                ac.set_compile_time(start.elapsed().as_secs());
                return ac;
            }
            let mut ac = self.build_ac(sol.epsilon(), &forced_by_propagation);
            ac.set_compile_time(start.elapsed().as_secs());
            ac
        } else {
            let mut discrepancy = 1;
            let mut complete_sol = None;
            let mut complete_ac = None;
            loop {
                let solution = self.do_discrepancy_iteration(discrepancy, 0.0);
                if self.parameters.start.elapsed().as_secs() < self.parameters.timeout || complete_sol.as_ref().is_none() {
                    complete_ac = Some(self.build_ac(solution.epsilon(), &forced_by_propagation));
                    complete_ac.as_mut().unwrap().set_compile_time(start.elapsed().as_secs());
                    //solution.print();
                    complete_sol = Some(solution);
                }
                if self.parameters.start.elapsed().as_secs() >= self.parameters.timeout || complete_sol.as_ref().unwrap().has_converged(self.parameters.epsilon) {
                    self.statistics.print();
                    complete_sol.unwrap().print();
                    return complete_ac.unwrap();
                }
                discrepancy += 1;
            }
        }        
    }

    pub fn build_ac(&self, epsilon: f64, forced_by_propagation:&(Vec<DistributionChoice>, Vec<UnconstrainedDistribution>)) -> Dac {
        let mut dac = Dac::new(epsilon);
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
        for (d, values) in forced_by_propagation.1.iter() {
            let sum_node = dac.sum_node(values.len());
            for (i, v) in values.iter().copied().enumerate() {
                let distribution_node = dac.distribution_value_node(&self.problem, *d, v);
                dac.add_input(sum_node.input_start() + i, distribution_node);
            }
            let sum_id = dac.add_node(sum_node);
            dac.add_input(child_id, sum_id);
            child_id += 1;
        }

        let mut map: FxHashMap<usize, NodeIndex> = FxHashMap::default();
        if has_node_search {
            let node_search = self.explore_cache(&mut dac, 0, &mut map);
            dac.add_input(child_id, node_search);
        }
        let root = dac.add_node(root);
        dac.set_root(root);
        dac
    }

    pub fn explore_cache(&self, dac: &mut Dac, cache_key_index: usize, c: &mut FxHashMap<usize, NodeIndex>) -> NodeIndex {
        if let Some(child_i) = c.get(&cache_key_index) {
            return *child_i;
        }
        let current = self.cache.get(&self.cache_keys[cache_key_index]).unwrap();
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
                let id = self.explore_cache(dac, cache_key, c);
                dac.add_input(child_id, id);
                child_id += 1;
            }
            let id = dac.add_node(prod_node);
            dac.add_input(sum_node.input_start() + sum_node_child, id);
            sum_node_child += 1;
        }
        let sum_index = dac.add_node(sum_node);
        c.insert(cache_key_index, sum_index);
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
                if self.problem[distribution].remaining(&self.state) != 1.0 {
                    let values = self.problem[distribution].iter_variables().filter(|v| !self.problem[*v].is_fixed(&self.state)).collect::<Vec<VariableIndex>>();
                    unconstrained_distribution_variables.push((distribution, values));
                }
            }
            
        }
        (forced_distribution_variables, unconstrained_distribution_variables)
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
    complete: bool,
}

impl CacheEntry {

    /// Returns a new cache entry
    pub fn new(bounds: Bounds, discrepancy: usize, distribution: Option<DistributionIndex>, children: FxHashMap<VariableIndex, CacheChildren>, cache_key_index: usize) -> Self {
        Self {
            bounds,
            discrepancy,
            distribution,
            children,
            cache_key_index,
            complete: false,
        }
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

    pub fn child_keys(&self, variable: VariableIndex) -> Vec<usize> {
        self.children.get(&variable).unwrap().children_keys.clone()
    }

    fn is_complete(&self) -> bool {
        self.complete
    }

    fn completed(&mut self) {
        self.complete = true;
    }
}

#[derive(Default, Clone)]
pub struct CacheChildren {
    children_keys: Vec<usize>,
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

    pub fn add_key(&mut self, key: usize) {
        self.children_keys.push(key);
    }
}

struct SearchResult {
    bounds: (Rational, Rational),
    cache_index: usize,
    complete: bool,
}
