//Schlandal
//Copyright (C) 2022-2023 A. Dubray
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


//! This module provides the main structure of the solver. It is responsible for orchestring
//! all the different parts and glue them together.
//! The algorithm starts by doing an initial propagation of the variables indentified during the
//! parsing of the input file, and then solve recursively the problem.
//! It uses the branching decision to select which variable should be propagated next, call the propagator
//! and identified the independent component.
//! It is also responsible for updating the cache and clearing it when the memory limit is reached.
//! Finally it save and restore the states of the reversible variables used in the solver.

use rustc_hash::FxHashMap;
use search_trail::{StateManager, SaveAndRestore};

use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::*;
use crate::heuristics::branching::BranchingDecision;
use crate::propagator::MixedPropagator;
use crate::search::statistics::Statistics;
use crate::common::*;
use crate::PEAK_ALLOC;

use rug::Float;
use rug::Assign;

/// Unit structure representing the the problem is UNSAT
#[derive(Debug)]
pub struct Unsat;

/// Type alias used for the solution of the problem, which is either a Float or UNSAT
type ProblemSolution = Result<Float, Unsat>;

type ProbaMassIn = Float;
type ProbaMassOut = Float;
type NodeSolution = (ProbaMassIn, ProbaMassOut, Float);

/// The solver for a particular set of Horn clauses. It is generic over the branching heuristic
/// and has a constant parameter that tells if statistics must be recorded or not.
pub struct Solver<'b, B, const S: bool>
where
    B: BranchingDecision + ?Sized,
{
    /// Implication graph of the input CNF formula
    graph: Graph,
    /// State manager that allows to retrieve previous values when backtracking in the search tree
    state: StateManager,
    /// Extracts the connected components in the graph
    component_extractor: ComponentExtractor,
    /// Heuristics that decide on which distribution to branch next
    branching_heuristic: &'b mut B,
    /// The propagator
    propagator: MixedPropagator,
    /// Cache used to store results of sub-problems
    cache: FxHashMap<CacheEntry, NodeSolution>,
    /// Statistics collectors
    statistics: Statistics<S>,
    /// Memory limit allowed for the solver. This is a global memory limit, not a cache-size limit
    mlimit: u64,
    /// The quality of the approximation. If p* is the true probability, the solver returns a probability p such that (p*/(1+e)) <= p <= (1+e)p*
    epsilon: f64,
    /// Vector used to store probability mass out of the distribution in each node
    distribution_out_vec: Vec<f64>,
}

impl<'b, B, const S: bool> Solver<'b, B, S>
where
    B: BranchingDecision + ?Sized,
{
    pub fn new(
        graph: Graph,
        state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: &'b mut B,
        propagator: MixedPropagator,
        mlimit: u64,
        epsilon: f64,
    ) -> Self {
        let cache = FxHashMap::default();
        let distribution_out_vec = vec![0.0; graph.number_distributions()];
        Self {
            graph,
            state,
            component_extractor,
            branching_heuristic,
            propagator,
            cache,
            statistics: Statistics::default(),
            mlimit,
            epsilon,
            distribution_out_vec,
        }
    }

    /// Returns the solution for the sub-problem identified by the component. If the solution is in
    /// the cache, it is not computed. Otherwise it is solved and stored in the cache.
    fn get_cached_component_or_compute(&mut self, component: ComponentIndex) -> NodeSolution {
        self.statistics.cache_access();
        let bit_repr = self.graph.get_bit_representation(&self.state, component, &self.component_extractor);
        match self.cache.get(&bit_repr) {
            None => {
                self.statistics.cache_miss();
                let f = self.choose_and_branch(component);
                self.cache.insert(bit_repr, f.clone());
                f
            },
            Some(f) => {
                f.clone()
            }
        }
    }

    /// Chooses a distribution to branch on using the heuristics of the solver and returns the
    /// solution of the component.
    /// The solution is the sum of the probability of the SAT children.
    fn choose_and_branch(&mut self, component: ComponentIndex) -> NodeSolution {
        let decision = self.branching_heuristic.branch_on(
            &self.graph,
            &self.state,
            &self.component_extractor,
            component,
        );
        if let Some(distribution) = decision {
            self.statistics.or_node();
            let mut p_in = f128!(0.0);
            let mut p_out = f128!(0.0);
            let mut p = f128!(0.0);
            let mut variables_to_branch = self.graph.distribution_variable_iter(distribution).filter(|v| !self.graph.is_variable_fixed(*v, &self.state)).collect::<Vec<VariableIndex>>();
            variables_to_branch.sort_unstable_by(|a, b| {
                let wa = self.graph.get_variable_weight(*a).unwrap();
                let wb = self.graph.get_variable_weight(*b).unwrap();
                wa.partial_cmp(&wb).unwrap()
            });
            for variable in variables_to_branch.iter().copied().rev() {
                if !self.graph.is_variable_fixed(variable, &self.state) {
                    let v_weight = self.graph.get_variable_weight(variable).unwrap();
                    self.state.save_state();
                    match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, component, &self.component_extractor) {
                        Err(_) => {
                            p_out += v_weight;
                        },
                        Ok(_) => {
                            let v = self.propagator.get_propagation_prob().clone();
                            if v != 0.0 {
                                let mut removed_proba = f128!(1.0);
                                let mut has_removed = false;
                                self.distribution_out_vec.fill(0.0);
                                for (d, variable, value) in self.propagator.assignments_iter().filter(|a| a.0 != distribution) {
                                    let weight = self.graph.get_variable_weight(variable).unwrap();
                                    if !value {
                                        has_removed = true;
                                        self.distribution_out_vec[d.0] += weight;
                                    }
                                }
                                if has_removed {
                                    for v in self.distribution_out_vec.iter().copied() {
                                        removed_proba *= 1.0 - v;
                                    }
                                }

                                let child_sol = self._solve(component);
                                p_in += child_sol.0 * &v;
                                p_out += child_sol.1 * &v + v_weight * (1.0 - removed_proba.clone());
                                p += child_sol.2 * &v; 
                                if let Some(proba) = self.approximate_count(p_in.clone(), p_out.clone()) {
                                    self.state.restore_state();
                                    return (p_in, p_out, proba);
                                }
                            } else {
                                p_out += v_weight;
                            }

                        }
                    };
                    self.state.restore_state();
                }
            }
            (p_in, p_out, p)
        } else {
            // The sub-formula is SAT, by definition return 1. In practice should not happen since in that case no components are returned in the detection
            debug_assert!(false);
            (f128!(1.0), f128!(0.0), f128!(1.0))
        }
    }

    /// Solves the problem for the sub-graph identified by component.
    pub fn _solve(&mut self, component: ComponentIndex) -> NodeSolution {
        // If the memory limit is reached, clear the cache.
        if PEAK_ALLOC.current_usage_as_mb() as u64 >= self.mlimit {
            self.cache.clear();
        }
        self.state.save_state();
        // Default solution with a probability/count of 1
        // Since the probability are multiplied between the sub-components, it is neutral. And if
        // there are no sub-components, this is the default solution.
        let mut p_in = f128!(1.0);
        let mut p_out = f128!(1.0);
        let mut p = f128!(1.0);
        // First we detect the sub-components in the graph
        if self
            .component_extractor
            .detect_components(&mut self.graph, &mut self.state, component, &mut self.propagator)
        {
            self.statistics.and_node();
            self.statistics
                .decomposition(self.component_extractor.number_components(&self.state));
            for sub_component in self.component_extractor.components_iter(&self.state) {
                let cache_solution = self.get_cached_component_or_compute(sub_component);
                p_in *= &cache_solution.0;
                p_out *= &cache_solution.1;
                p *= &cache_solution.2;
                if p == 0.0 {
                    break;
                }
            }
        } else {
            p_out.assign(0.0);
        }
        self.state.restore_state();
        (p_in, p_out, p)
    }
    
    fn solve_by_search(&mut self) -> ProblemSolution {
        // First set the number of clause in the propagator. This can not be done at the initialization of the propagator
        // because we need it to parse the input file as some variables might be detected as always being true or false.
        self.propagator.set_number_clauses(self.graph.number_clauses());
        // Doing an initial propagation to detect some UNSAT formula from the start
        match self.propagator.propagate(&mut self.graph, &mut self.state, ComponentIndex(0), &self.component_extractor) {
            Err(_) => ProblemSolution::Err(Unsat),
            Ok(_) => {
                let mut p_in = f128!(1.0);
                let mut p_out = f128!(1.0);
                
                for distribution in self.graph.distributions_iter() {
                    if self.graph.distribution_number_false(distribution, &self.state) > 0 {
                        let mut sum_neg = 0.0;
                        for variable in self.graph.distribution_variable_iter(distribution) {
                            if let Some(v) = self.graph.get_variable_value(variable, &self.state) {
                                if v {
                                    p_in *= self.graph.get_variable_weight(variable).unwrap();
                                } else {
                                    sum_neg += self.graph.get_variable_weight(variable).unwrap();
                                }
                            }
                        }
                        p_out *= 1.0 - sum_neg;
                    }
                }
                p_out = 1.0 - p_out;
                if p_in == 1.0 {
                    self.branching_heuristic.init(&self.graph, &self.state);
                    let solution = self._solve(ComponentIndex(0));
                    self.statistics.print();
                    ProblemSolution::Ok(solution.2)
                } else {
                    match self.approximate_count(p_in.clone(), p_out.clone()) {
                        None => {
                            self.branching_heuristic.init(&self.graph, &self.state);
                            let mut solution = self._solve(ComponentIndex(0));
                            solution.2 *= p_in;
                            self.statistics.print();
                            ProblemSolution::Ok(solution.2)
                        },
                        Some(proba) => ProblemSolution::Ok(proba),
                    }
                }
            }
        }
    }
    
    fn approximate_count(&mut self, p_in: Float, p_out: Float) -> Option<Float> {
        let lb = p_in ;
        let ub: Float = 1.0 - p_out;
        if ub <= 0.0 {
            return Some(f128!(0.0));
        }
        if ub <= lb.clone()*(1.0 + self.epsilon).powf(2.0) {
            let approximation = f128!((lb*&ub).sqrt());
            Some(approximation)
        } else {
            None
        }
    }
    
    /// Solve the problems represented by the graph with the given branching heuristic.
    /// It finds all the assignments to the probabilistic variables for which there
    /// exists an assignment to the deterministic variables that respect the constraints.
    /// Each assignment is weighted by the product of the probabilistic variables assigned to true.
    pub fn solve(&mut self) -> Option<Float> {
        match self.solve_by_search() {
            Ok(p) => {
                Some(p)
            }
            Err(_) => {
                None
            }
        }
    }
}