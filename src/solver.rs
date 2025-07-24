use rustc_hash::FxHashMap;
use search_trail::{SaveAndRestore, StateManager};

use crate::logger::Logger;
use crate::branching::BranchingDecision;
use crate::common::*;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::problem::{DistributionIndex, Problem, VariableIndex};
use crate::target::ac::*;
use crate::target::*;
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use crate::PEAK_ALLOC;
use crate::caching::CacheKey;
use crate::args::Args;
use malachite::rational::Rational;
use std::time::Instant;

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
pub struct Solver<const S: bool> {
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
    /// Cache for the sub-problems solved
    cache: FxHashMap<CacheKey, NodeIndex>,
    /// Statistics gathered during the solving
    statistics: Logger<S>,
}

impl<const S: bool> Solver<S> {
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
        }
    }

    /// Restores the state of the solver to the previous state
    fn restore(&mut self) {
        self.propagator.restore(&self.state);
        self.state.restore_state();
    }

    /// Solves the problem represented by this solver using a DPLL-search based method.
    pub fn compute_pwmc(&mut self, parameters: &SolverParameters) -> Solution {
        let mut ac = Ac::default();
        if let Some(sol) = self.preprocess(parameters) {
            self.statistics.print();
            return sol;
        }

        let root_model = ac.prod_node();

        // Adds everything propagated to the circuit
        //
        // First, we consider the circuit for the model count
        {
            // The weighted model count of the search tree must be multiplied by
            //  1. Everything set to true during the pre-processing
            //  2. Every unconstrained distribution
            for assignment in self.propagator.assignments_iter(&self.state) {
                let variable = assignment.to_variable();
                if assignment.is_positive() && self.problem[variable].is_probabilitic() {
                    let distribution = self.problem[variable].distribution().unwrap();
                    let node = ac.get_distribution_node(distribution, variable, self.problem[variable].weight().unwrap());
                    ac.add_edge(root_model, node);
                }
            }

            if self.component_extractor.detect_components(&mut self.problem, &mut self.state, ComponentIndex(0)) {
                // A number of distribution are not fixed but do not appear in the
                // sub-components, we can compute their contribution in closed form
                for distribution in self.component_extractor
                    .component_removed_distribution_iter(ComponentIndex(0))
                    .filter(|d| self.problem[*d].is_constrained(&self.state) && self.problem[*d].is_partial_domain(&self.state)) {
                        let sum_distribution_node = self.sum_node_distribution_partial_domain(&mut ac, distribution);
                        ac.add_edge(root_model, sum_distribution_node);
                }
            }
        }

        self.restructure_after_preprocess();

        if self.problem.number_clauses() > 0 {
            self.do_discrepancy_iteration(&mut ac, root_model, usize::MAX, parameters);
            /* TODO
            if !parameters.lds {
            } else {
                self.do_discrepancy_iteration(&mut ac, root_model, usize::MAX, parameters);
                let mut discrepancy = 1;
                loop {
                    self.do_discrepancy_iteration(&mut ac, root_model, discrepancy, parameters);
                    discrepancy += 1;
                }
            }
            */
        }
        self.statistics.print();
        println!("AC size: {} nodes {} edges", ac.number_nodes(), ac.number_edges());
        //println!("{}", ac.to_graphviz());
        Solution::new(ac[root_model].value(), ac[root_model].value(), parameters.start.elapsed().as_secs())
    }

    /// Preprocess the problem, if the problem is solved during the preprocess, return a solution.
    /// Returns None otherwise
    fn preprocess(&mut self, parameters: &SolverParameters) -> Option<Solution> {
        self.propagator.init(self.problem.number_clauses());
        let mut preprocessor = Preprocessor::new(
            &mut self.problem,
            &mut self.state,
            &mut self.propagator,
            &mut self.component_extractor,
        );
        let preproc = preprocessor.preprocess();
        if preproc.is_err() {
            return Some(Solution::new(
                rational(0.0),
                rational(1.0),
                parameters.start.elapsed().as_secs(),
            ));
        }
        None
    }

    fn restructure_after_preprocess(&mut self) {
        self.problem.clear_after_preprocess(&mut self.state);
        self.component_extractor.shrink(
            self.problem.number_clauses(),
            self.problem.number_variables(),
            self.problem.number_distributions(),
        );
        self.propagator.reduce(
            self.problem.number_clauses(),
            self.problem.number_variables(),
            &mut self.state
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

    pub fn do_discrepancy_iteration(&mut self, ac: &mut Ac, root: NodeIndex, discrepancy: usize, parameters: &SolverParameters) {
        self.pwmc(ac, root, ComponentIndex(0), 1, discrepancy, parameters);
    }

    fn pwmc(&mut self, ac: &mut Ac, parent: NodeIndex, component: ComponentIndex, level: isize, discrepancy: usize, parameters: &SolverParameters) {
        self.statistics.or_node();
        if PEAK_ALLOC.current_usage_as_mb() as u64 >= parameters.memory_limit {
            self.cache.clear();
        }
        let cache_key = self.component_extractor[component].get_cache_key();
        self.statistics.cache_access();
        let current_node = self.cache.remove(&cache_key).unwrap_or_else(|| {
            self.statistics.cache_miss();
            let node = ac.sum_node();
            ac[node].set_distribution(self.branching_heuristic.branch_on(&self.problem, &mut self.state, &self.component_extractor, component));
            ac[node].incomplete();
            node
        });
        if ac[current_node].is_complete() {
            ac.add_edge(parent, current_node);
            return;
        }
        // We are sure that the node has a distribution to branch on, otherwise no components are
        // detected.
        let distribution = ac[current_node].distribution().unwrap();
        let mut child_id = 0;
        let mut complete = true;
        for variable in self.problem[distribution].iter_variables() {
            if self.problem[variable].is_fixed(&self.state) {
                continue
            }
            if parameters.start.elapsed().as_secs() >= parameters.timeout || child_id == discrepancy {
                complete = false;
                break;
            }
            if parameters.approx_subproblems {
                panic!("Approx sub-problems not yet implemented in new version");
            }
            self.state.save_state();
            match self.propagator.propagate_variable(variable, true, &mut self.problem, &mut self.state, component, &mut self.component_extractor, level) {
                Err(_) => {
                    self.statistics.unsat();
                    // TODO
                },
                Ok(_) => {
                    // The propagation has not detected any UNSAT, we create the sub-circuit for
                    // the sub-problems.
                    // We detect independent components at this step; hence, all sub-circuits are
                    // linked with a product node.
                    let child_node = ac.prod_node();
                    // Then, we also create a sub-circuit for the values propagated to true
                    // during propagation
                    for literal in self.propagator.assignments_iter(&self.state).filter(|l| l.is_positive() && self.problem[l.to_variable()].is_probabilitic()) {
                        let variable = literal.to_variable();
                        let distribution = self.problem[variable].distribution().unwrap();
                        let node = ac.get_distribution_node(distribution, variable, self.problem[variable].weight().unwrap());
                        ac.add_edge(child_node, node);
                    }
                    self.state.save_state();
                    if self.component_extractor.detect_components(&mut self.problem, &mut self.state, component) {
                        // A number of distribution are not fixed but do not appear in the
                        // sub-components, we can compute their contribution in closed form
                        for distribution in self
                            .component_extractor
                            .component_removed_distribution_iter(component)
                            .filter(|d| self.problem[*d].is_constrained(&self.state) && self.problem[*d].is_partial_domain(&self.state)) {
                            let sum_distribution_node = self.sum_node_distribution_partial_domain(ac, distribution);
                            ac.add_edge(child_node, sum_distribution_node);
                        }
                        for sub_component in self.component_extractor.components_iter(&self.state) {
                            // The recursive call to pwmc will add the link between the child_node
                            // and the node for the sub-problem.
                            self.pwmc(ac, child_node, sub_component, level + 1, discrepancy, parameters);
                            if ac[child_node].value() == 0.0 {
                                complete = true;
                                break;
                            }
                            complete &= ac[child_node].is_complete();
                        }
                    }
                    self.restore();
                    ac.add_edge(current_node, child_node);
                }
            };
            self.restore();
            child_id += 1;
        }
        if complete {
            ac[current_node].complete();
        }
        self.cache.insert(cache_key, current_node);
        ac.add_edge(parent, current_node);
    }

    fn sum_node_distribution_partial_domain(&self, ac: &mut Ac, distribution: DistributionIndex) -> NodeIndex {
        let node = ac.sum_node();
        for variable in self.problem[distribution].iter_variables().filter(|v| !self.problem[*v].is_fixed(&self.state)) {
            let child = ac.get_distribution_node(distribution, variable, self.problem[variable].weight().unwrap());
            ac.add_edge(node, child);
        }
        node
    }

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
