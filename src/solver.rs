use rustc_hash::FxHashMap;
use search_trail::{SaveAndRestore, StateManager};

use crate::logger::Logger;
use crate::branching::BranchingDecision;
use crate::common::*;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::problem::{DistributionIndex, Problem, VariableIndex};
use crate::ac::ac::{NodeIndex, Dac};
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use crate::PEAK_ALLOC;
use crate::caching::CacheKey;
use crate::cache::*;
use crate::args::Args;
use malachite::rational::Rational;
use std::time::Instant;

pub type DistributionChoice = (DistributionIndex, VariableIndex);
pub type DistributionPartialDomain = (DistributionIndex, Vec<VariableIndex>);

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
    statistics: Logger<S>,
    /// Product of the weight of the variables set to true during propagation
    preproc_in: Option<Rational>,
    /// Probability of removed interpretation during propagation
    preproc_out: Option<Rational>,
    /// The caches present in the cache. Used during compilation to reconstruct the AC from the
    /// cache (follow the children of a node)
    cache_keys: Vec<CacheKey>,
}

impl<const S: bool, const C: bool> Solver<S, C> {
    pub fn new(
        problem: Problem,
        state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: Box<dyn BranchingDecision>,
        propagator: Propagator,
    ) -> Self {
        Self {
            problem,
            state,
            component_extractor,
            branching_heuristic,
            propagator,
            cache: FxHashMap::default(),
            statistics: Logger::default(),
            preproc_in: None,
            preproc_out: None,
            cache_keys: vec![],
        }
    }

    /// Restores the state of the solver to the previous state
    fn restore(&mut self) {
        self.propagator.restore(&self.state);
        self.state.restore_state();
    }

    /// Solves the problem represented by this solver using a DPLL-search based method.
    pub fn search(&mut self, parameters: &SolverParameters) -> Solution {
        let max = self.problem.distributions_iter().map(|d| rational(self.problem[d].remaining(&self.state))).product::<Rational>();
        if let Some(sol) = self.preprocess(&max, parameters) {
            return sol;
        }
        self.restructure_after_preprocess();

        if self.problem.number_clauses() == 0 {
            let lb = self.preproc_in.clone().unwrap();
            let ub = max - self.preproc_out.clone().unwrap();
            return Solution::new(lb, ub, parameters.start.elapsed().as_secs(), true);
        }
        if !parameters.lds {
            let sol = self.do_discrepancy_iteration(usize::MAX, parameters.epsilon, parameters);
            self.statistics.peak_memory(PEAK_ALLOC.peak_usage_as_mb());
            self.statistics.lower_bound(sol.bounds().0);
            self.statistics.upper_bound(sol.bounds().1);
            self.statistics.print();
            sol
        } else {
            let mut discrepancy = 1;
            let mut complete_sol = None;
            loop {
                let solution = self.do_discrepancy_iteration(discrepancy, 0.0, parameters);
                if solution.epsilon() < 0.01 {
                    discrepancy = usize::MAX;
                } else {
                    discrepancy += 1;
                }
                if parameters.start.elapsed().as_secs() < parameters.timeout || complete_sol.as_ref().is_none() {
                    solution.print();
                    complete_sol = Some(solution);
                }
                if parameters.start.elapsed().as_secs() >= parameters.timeout || complete_sol.as_ref().unwrap().has_converged(parameters.epsilon) {
                    self.statistics.peak_memory(PEAK_ALLOC.peak_usage_as_mb());
                    self.statistics.print();
                    return complete_sol.unwrap()
                }
            }
        }
    }

    /// Preprocess the problem, if the problem is solved during the preprocess, return a solution.
    /// Returns None otherwise
    fn preprocess(&mut self, max: &Rational, parameters: &SolverParameters) -> Option<Solution> {
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
                parameters.start.elapsed().as_secs(),
                true,
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

    pub fn do_discrepancy_iteration(&mut self, discrepancy: usize, eps: f64, parameters: &SolverParameters) -> Solution {
        let result = self.pwmc(ComponentIndex(0), 1, discrepancy, eps, parameters);
        let p_in = result.bounds.0.clone();
        let p_out = result.bounds.1.clone();
        let lb = p_in * self.preproc_in.clone().unwrap();
        let ub: Rational = rational(1.0) - (self.preproc_out.clone().unwrap() + p_out * self.preproc_in.clone().unwrap());
        Solution::new(lb, ub, parameters.start.elapsed().as_secs(), result.complete)
    }

    fn pwmc(&mut self, component: ComponentIndex, level: isize, discrepancy: usize, eps: f64, parameters: &SolverParameters) -> SearchResult {
        if PEAK_ALLOC.current_usage_as_mb() as u64 >= parameters.memory_limit {
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
            let component_all_distributions = if C { self.get_all_distributions_for_ac(component, None) } else { vec![] };
            CacheEntry::new(
                cache_key_index,
                component_all_distributions)
        });
        if cache_entry.distribution().is_none() {
            self.statistics.or_node();
            cache_entry.set_distribution(self.branching_heuristic.branch_on(&self.problem, &mut self.state, &self.component_extractor, component));
        }

        let mut complete = cache_entry.distribution().is_some();
        if cache_entry.discrepancy() < discrepancy && !cache_entry.is_complete() {
            // New values for the count of satisfying and unsatisfying assignments
            let mut new_p_in = rational(0.0);
            let mut new_p_out = rational(0.0);
            // Distribution to branch on
            let distribution = cache_entry.distribution().unwrap();

            // Maximum probablity that the component can have
            let max_probability = self.component_extractor.component_distribution_iter(component).map(|d| {
                rational(self.problem[d].remaining(&self.state))
            }).product::<Rational>();
            // If a branching gives an UNSAT, this value must be added to the p_out count
            let unsat_factor = max_probability.clone() / rational(self.problem[distribution].remaining(&self.state));

            let mut child_id = 0;
            for variable in self.problem[distribution].iter_variables() {
                if self.problem[variable].is_fixed(&self.state) {
                    continue;
                }
                if parameters.start.elapsed().as_secs() >= parameters.timeout || child_id == discrepancy {
                    complete = false;
                    break;
                }
                // If subproblems are approximated, we check before branching if enough probability
                // mass has been accumulated
                if parameters.approx_subproblems {
                    let ub = max_probability.clone() - new_p_out.clone();
                    let lb = new_p_in.clone();
                    // See CP24 paper for explanation on the formula. Basically, the bounds are
                    // close (w.r.t. the provided epsilon) enough to approximate.
                    if ub <= lb*rational((1.0 + eps)*(1.0 + eps)) {
                        complete = true;
                        break;
                    }
                }
                let v_weight = self.problem[variable].weight().unwrap();
                self.state.save_state();
                // Performs the branching
                match self.propagator.propagate_variable(variable, true, &mut self.problem, &mut self.state, component, &mut self.component_extractor, level) {
                    Err(_) => {
                        self.statistics.unsat();
                        new_p_out += v_weight * unsat_factor.clone();
                        if C {
                            // If the sub-problem is unsat, then all remaining domains are empty
                            // (no solution exist)
                            cache_entry.add_child(variable, None);
                        }
                    },
                    Ok(_) => {
                        let p = self.propagator.get_propagation_prob();
                        let removed = unsat_factor.clone() - self.component_extractor
                            .component_distribution_iter(component)
                            .filter(|d| *d != distribution)
                            .map(|d| rational(self.problem[d].remaining(&self.state)))
                            .product::<Rational>();
                        new_p_out += removed * v_weight;

                        // Creates the entry for the children of the current sub-problem (i.e., a
                        // vector of sub-problems for each independent component resulting from the
                        // propagation).
                        let mut child_entry: Vec<usize> = vec![];

                        // Decomposing into independent components
                        let mut prod_p_in = rational(1.0);
                        let mut prod_p_out = rational(1.0);
                        let prod_maximum_probability = self.component_extractor
                            .component_distribution_iter(component)
                            .filter(|d| self.problem[*d].is_constrained(&self.state))
                            .map(|d| rational(self.problem[d].remaining(&self.state)))
                            .product::<Rational>();

                        self.state.save_state();
                        if self.component_extractor.detect_components(&mut self.problem, &mut self.state, component) {
                            self.statistics.decomposition(self.component_extractor.number_components(&self.state));
                            let number_components = self.component_extractor.number_components(&self.state);
                            let new_eps = eps.powf(1.0 / number_components as f64);
                            for sub_component in self.component_extractor.components_iter(&self.state) {
                                let sub_maximum_probability = self.component_extractor[sub_component].max_probability();
                                let sub_solution = self.pwmc(sub_component, level + 1, discrepancy - child_id, new_eps, parameters);
                                if !sub_solution.complete {
                                    complete = false;
                                }
                                prod_p_in *= &sub_solution.bounds.0;
                                prod_p_out *= sub_maximum_probability - &sub_solution.bounds.1;
                                if prod_p_in == 0.0 {
                                    complete = true;
                                    break;
                                }
                                if C {
                                    child_entry.push(sub_solution.cache_index);
                                }
                            }
                        }
                        if C {
                            if prod_p_in > 0.0 {
                                cache_entry.add_child(variable, Some(child_entry));
                            } else {
                                cache_entry.add_child(variable, None);
                            }
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
            cache_entry.set_discrepancy(discrepancy);
            cache_entry.set_bounds((new_p_in, new_p_out));
        }
        if complete {
            cache_entry.completed();
        }
        let result = SearchResult {
            bounds: cache_entry.bounds().clone(),
            cache_index: cache_entry.cache_key_index(),
            complete,
        };
        self.cache.insert(cache_key, cache_entry);
        result
    }
}

impl<const S: bool, const C: bool> Solver<S, C> {

    pub fn compile(&mut self, parameters: &SolverParameters) -> Dac {
        if !C {
            panic!("Calling the compile function with a search-based instantiation of the solver");
        }

        let max = self.problem.distributions_iter().map(|d| rational(self.problem[d].remaining(&self.state))).product::<Rational>();

        if self.preprocess(&max, parameters).is_none() {
            return Dac::unsat();
        }

        let mut ac = Dac::new(true);
    
        // First, all the things assigned during pre-processing are put into the AC.
        // This includes two things for both (model and non-model):
        //  - All variables assigned to T
        //  - All unconstrained distribution
        //
        // These two elements are binded using a product node and unconstrained distributions are
        // summed.
        let assigned_variables = self.propagator
            .assignments_iter(&self.state)
            .filter(|l| self.problem[l.to_variable()].is_probabilitic() && l.is_positive())
            .map(|l| {
                let var = l.to_variable();
                let dist = self.problem[var].distribution().unwrap();
                (dist, var)
            })
            .collect::<Vec<(DistributionIndex, VariableIndex)>>();

        let unconstrained_distributions = self.propagator
            .unconstrained_distributions_iter()
            .filter(|d| self.problem[*d].remaining(&self.state) != 1.0)
            .map(|d| {
                let vs = self.problem[d].iter_variables().filter(|v| !self.problem[*v].is_fixed(&self.state)).collect::<Vec<VariableIndex>>();
                ac.sum_distribution_node(&self.problem, d, &vs)
            }).collect::<Vec<NodeIndex>>();

        let prod_preproc_node = ac.prod_node(assigned_variables.len() + unconstrained_distributions.len());
        for (child_id, (dist, var)) in assigned_variables.iter().copied().enumerate() {
            let input_node = ac.distribution_value_node(&self.problem, dist, var);
            ac.add_input(child_id, &prod_preproc_node, input_node);
        }

        // We can remove unused data from the problem
        self.restructure_after_preprocess();
        let number_root_children = if self.problem.number_clauses() == 0 { 1 } else { 2 };
        let root_model = ac.prod_node(number_root_children);
        let root_non_model = ac.sum_node(number_root_children);

        for (child_id, input_node) in unconstrained_distributions.iter().copied().enumerate() {
            ac.add_input(child_id + assigned_variables.len(), &prod_preproc_node, input_node);
        }
        let prod_preproc_node_index = ac.add_node(prod_preproc_node);
        ac.add_input(0, &root_model, prod_preproc_node_index);
        ac.add_input(0, &root_non_model, prod_preproc_node_index);


        // Additionaly, for the non-model, there are assignments detected as non-model at the root
        // (i.e., all models containing variables set to false during the preprocessing).
        let sub_node = ac.sub_node(2);
        let max_node = ac.constant_node(max);
        // We need to remove from the max the product of the remaining distribution sums
        let p_node = ac.prod_node(self.problem.number_distributions());
        for (idx, d) in self.problem.distributions_iter().enumerate() {
            let variables = self.problem[d].iter_variables().collect::<Vec<VariableIndex>>();
            let child = ac.sum_distribution_node(&self.problem, d, &variables);
            ac.add_input(idx, &p_node, child);
        }
        let p_index = ac.add_node(p_node);
        ac.add_input(0, &sub_node, max_node);
        ac.add_input(1, &sub_node, p_index);

        if self.problem.number_clauses() != 0 {
            if !parameters.lds {
                self.do_discrepancy_iteration(usize::MAX, parameters.epsilon, parameters);
                self.statistics.print();
                let n = self.build_ac(&mut ac);
                ac.add_input(1, &root_model, n);
            } else {
                let mut discrepancy = 1;
                loop {
                    let solution = self.do_discrepancy_iteration(discrepancy, 0.0, parameters);
                    if parameters.start.elapsed().as_secs() >= parameters.timeout || solution.is_exact() {
                        self.statistics.print();
                        let n = self.build_ac(&mut ac);
                        let p = ac.prod_node(2);
                        ac.add_input(0, &p, prod_preproc_node_index);
                        ac.add_input(1, &p, n);
                        let pidx = ac.add_node(p);
                        ac.add_input(1, &root_non_model, pidx);
                        break;
                    }
                    discrepancy += 1;
                }
            }        
        }
        let rid = ac.add_node(root_model);
        ac.set_root_model(rid);
        let rid = ac.add_node(root_non_model);
        ac.set_root_non_model(rid);
        ac
    }

    pub fn build_ac(&self, ac: &mut Dac) -> NodeIndex {
        let mut map: FxHashMap<usize, NodeIndex> = FxHashMap::default();
        self.explore_cache(ac, 0, &mut map)
    }

    pub fn explore_cache(&self, ac: &mut Dac, cache_key_index: usize, c: &mut FxHashMap<usize, NodeIndex>) -> (Option<NodeIndex>, Option<NodeIndex>) {
        if let Some(child_i) = c.get(&cache_key_index) {
            return *child_i;
        }

        let current = self.cache.get(&self.cache_keys[cache_key_index]).unwrap();
        let mut children_model: Vec<NodeIndex> = vec![];
        let mut children_non_model: Vec<NodeIndex> = vec![];

        let parent_domains = current.domains();

        // Iterate on the variables the distribution with the associated cache key
        for variable in current.children_variables() {
            let variable_distribution = self.problem[variable].distribution().unwrap();
            match current.child_keys(variable) {
                None => {
                    // The subproblem is UNSAT when branching on variable. All the probability mass
                    // goes to the unsat root
                    let node = ac.prod_node(2);
                    let vweight = ac.distribution_value_node(&self.problem, variable_distribution, variable);
                    let subproblem = self.product_distributions(ac, parent_domains, |d| d != variable_distribution);
                    ac.add_input(0, &node, vweight);
                    ac.add_input(1, &node, subproblem);
                    children_non_model.push(ac.add_node(node));
                },
                Some(children) => {
                    // Each children is SAT, we compute the sub-circuits.
                    // 
                    let node_model = ac.prod_node(1 + children.len());
                    let node_non_model = ac.sub_node(2);
                    // The difference between the domains before and after give the probability
                    // mass that must be added to the non_model circuit
                    let mut domains: Vec<DistributionPartialDomain> = vec![];
                    for child in children.iter().copied() {
                        let child_entry = self.cache.get(&self.cache_keys[child]).unwrap();
                        domains.append(&mut child_entry.domains().clone());
                    }
                    // First, we compute the sub-circuit for the values propagated at true and the
                    // unconstrained distributions.
                    let circuit_propagated = self.get_circuit_propagated_values(ac, &parent_domains, &domains, variable_distribution);
                    // Then, we get the removed probability mass from the propagation
                    let removed_probability_mass = self.get_circuit_mass_non_model_by_propagation(ac, &parent_domains, &domains, |d| d != variable_distribution);

                    // Then, we can recursively explore all children and compute their sub-circuit
                    let mut circuit_subproblem_model: Vec<NodeIndex> = vec![];
                    let mut circuit_subproblem_non_model: Vec<NodeIndex> = vec![];
                    for child in children.iter().copied() {
                        let (child_model, child_non_model) = self.explore_cache(ac, child, c);
                        if let Some(child) = child_model {
                            circuit_subproblem_model.push(child);
                        }
                        if let Some(child) = child_non_model {
                            circuit_subproblem_non_model.push(child);
                        }
                    }
                    // For the models, we just take the product of the sub-circuits
                    for (i, child) in circuit_subproblem_model.iter().copied().enumerate() {
                        ac.add_input(1 + i, &node_model, child);
                    }

                    let n1 = ac.get_input(removed_probability_mass, 0);
                    ac.add_input(0, &node_non_model, n1);
                    let n2 = ac.prod_node(circuit_subproblem_non_model.len());
                    for (child_id, child) in children.iter().copied().enumerate() {
                        let child_domains = self.cache.get(&self.cache_keys[child]).unwrap().domains();
                        let child_node = ac.sub_node(2);
                        let max = self.product_distributions(ac, &child_domains, |d| true);
                        ac.add_input(0, &child_node, max);
                        ac.add_input(1, &child_node, circuit_subproblem_non_model[child_id]);
                        let n = ac.add_node(child_node);
                        ac.add_input(child_id, &n2, n);
                    }
                    let n2 = ac.add_node(n2);
                    ac.add_input(0, &node_non_model, n1);
                    ac.add_input(1, &node_non_model, n2);
                    
                    children_model.push(ac.add_node(node_model));
                    children_non_model.push(ac.add_node(node_non_model));
                },
            };
        }
        let root_model = ac.sum_node(children_model.len());
        for (child_id, child) in children_model.iter().copied().enumerate() {
            ac.add_input(child_id, &root_model, child);
        }
        let root_model = ac.add_node(root_model);
        let root_non_model = ac.sum_node(children_non_model.len());
        for (child_id, child) in children_model.iter().copied().enumerate() {
            ac.add_input(child_id, &root_non_model, child);
        }
        let root_non_model = ac.add_node(root_non_model);
        //c.insert(cache_key_index, sum_index);
        (Some(root_model), Some(root_non_model))
    }

    fn product_distributions<F>(&self, ac: &mut Dac, domains: &[DistributionPartialDomain], filter: F)-> NodeIndex
        where F: Fn(DistributionIndex) -> bool
    {
        let node = ac.prod_node(domains.len());
        for (child_id, domain) in domains.iter().filter(|domain| filter(domain.0)).enumerate() {
            let child = ac.sum_distribution_node(&self.problem, domain);
            ac.add_input(child_id, &node, child);
        }
        ac.add_node(node)
    }

    fn get_circuit_mass_non_model_by_propagation<F>(&self, ac: &mut Dac, domains_parent: &[DistributionPartialDomain], domains_child: &[DistributionPartialDomain], filter: F) -> NodeIndex
        where F: Fn(DistributionIndex) -> bool
    {
        // TODO optimise this sub-circuit to factorized the distributions whose domain do not
        // change
        let node = ac.sub_node(2);
        let child_left = self.product_distributions(ac, domains_parent, &filter);
        let child_right = self.product_distributions(ac, domains_child, &filter);
        ac.add_input(0, &node, child_left);
        ac.add_input(1, &node, child_right);
        ac.add_node(node)
    }

    fn get_circuit_propagated_values(&self, ac: &mut Dac, domains_parent: &[DistributionPartialDomain], domains_child: &[DistributionPartialDomain], skip: DistributionIndex) -> NodeIndex {
        let mut nodes: Vec<NodeIndex> = vec![];
        let mut i = 0;
        let mut j = 0;
        while i < domains_parent.len() && j < domains_child.len() {
            while domains_parent[i].0 != domains_child[j].0 {
                if domains_parent[i].0 == skip {
                    i += 1;
                    continue;
                }
                let node = ac.sum_distribution_node(&self.problem, &domains_parent[i]);
                nodes.push(node);
                i += 1;
            }
            j += 1;
        }
        let prod = ac.prod_node(nodes.len());
        for (child_index, child) in nodes.iter().copied().enumerate() {
            ac.add_input(child_index, &prod, child);
        }
        ac.add_node(prod)
    }

    /// Returns the choices (i.e., assignments to the distributions) made during the propagation as
    /// well as the distributions that are not constrained anymore.
    /// A choice for a distribution is a pair (d, i) = (DistributionIndex, usize) that indicates that
    /// the i-th value of disitribution d is true.
    /// An unconstrained distribution is a pair (d, v) = (DistributionIndex, Vec<usize>) that
    /// indicates that distribution d does not appear in any clauses and its values in v are not
    /// set yet.
    fn domain_fixed_or_unconstrained(&mut self) -> Vec<DistributionPartialDomain> {
        let mut domains: Vec<DistributionPartialDomain> = vec![];

        if self.propagator.has_assignments(&self.state) || self.propagator.has_unconstrained_distribution() {
            // First, we look at the assignments
            for literal in self.propagator.assignments_iter(&self.state) {
                let variable = literal.to_variable();
                // Only take probabilistic variables set to true
                if self.problem[variable].is_probabilitic() && literal.is_positive() && self.problem[variable].weight().unwrap() != 1.0 {
                    let distribution = self.problem[variable].distribution().unwrap();
                    // This represent which "probability index" is send to the node
                    domains.push((distribution, vec![variable]));
                }
            }

            // Then, for each unconstrained distribution, we create a sum_node, but only if the
            // distribution has at least one value set to false.
            // Otherwise it would always send 1.0 to the product node.
            for distribution in self.propagator.unconstrained_distributions_iter() {
                if self.problem[distribution].remaining(&self.state) != 1.0 {
                    let values = self.problem[distribution].iter_variables().filter(|v| !self.problem[*v].is_fixed(&self.state)).collect::<Vec<VariableIndex>>();
                    domains.push((distribution, values));
                }
            }
            
        }
        domains
    }

    fn get_all_distributions_for_ac(&self, component: ComponentIndex, filter_current: Option<DistributionIndex>) -> Vec<(DistributionIndex, Vec<VariableIndex>)> {
        let mut v = vec![];
        for distribution in self.component_extractor
            .component_distribution_iter(component)
            .filter(|d| (filter_current.is_none() || filter_current.unwrap() != *d) && self.problem[*d].is_constrained(&self.state) && self.problem[*d].remaining(&self.state) != 1.0) {
                let remaining_variables = self.problem[distribution].iter_variables().filter(|v| !self.problem[*v].is_fixed(&self.state)).collect::<Vec<VariableIndex>>();
                v.push((distribution, remaining_variables));
        }
        v
    }
}

struct SearchResult {
    bounds: (Rational, Rational),
    cache_index: usize,
    complete: bool,
}

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
    /// If true, perform LDS
    lds: bool,
    /// If true, approximate sub-problems to have a epsilon-approximation at the root
    approx_subproblems: bool,
}

impl SolverParameters {

    pub fn new(args: &Args) -> Self {
        Self {
            memory_limit: args.memory(),
            epsilon: args.epsilon(),
            timeout: args.timeout(),
            start: Instant::now(),
            lds: args.lds(),
            approx_subproblems: args.approx_subproblems(),
        }
    }
}
