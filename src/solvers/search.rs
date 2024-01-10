//Schlandal
//Copyright (C) 2022-2023 A. Dubray, L. Dierckx
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
use crate::branching::BranchingDecision;
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use super::statistics::Statistics;
use crate::common::*;
use crate::PEAK_ALLOC;

use rug::Float;
use rug::Assign;

use super::*;


/// The solver for a particular set of Horn clauses. It is generic over the branching heuristic
/// and has a constant parameter that tells if statistics must be recorded or not.
pub struct SearchSolver<B, const S: bool>
where
    B: BranchingDecision
{
    /// Implication graph of the input CNF formula
    graph: Graph,
    /// State manager that allows to retrieve previous values when backtracking in the search tree
    state: StateManager,
    /// Extracts the connected components in the graph
    component_extractor: ComponentExtractor,
    /// Heuristics that decide on which distribution to branch next
    branching_heuristic: Box<B>,
    /// The propagator
    propagator: Propagator,
    /// Cache used to store results of sub-problems
    cache: FxHashMap<CacheEntry, Bounds>,
    /// Statistics collectors
    statistics: Statistics<S>,
    /// Memory limit allowed for the solver. This is a global memory limit, not a cache-size limit
    mlimit: u64,
    /// Vector used to store probability mass out of the distribution in each node
    distribution_out_vec: Vec<f64>,
    epsilon: f64,
    preproc_in: Float,
    preproc_out: Float,
    prefix_factor: Float,
}

impl<B, const S: bool> SearchSolver<B, S>
where
    B: BranchingDecision
{
    pub fn new(
        graph: Graph,
        state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: Box<B>,
        propagator: Propagator,
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
            distribution_out_vec,
            epsilon,
            preproc_in: f128!(0.0),
            preproc_out: f128!(1.0),
            prefix_factor: f128!(1.0),
        }
    }
    
    fn restore(&mut self) {
        self.propagator.restore(&self.state);
        self.state.restore_state();
    }

    /// Returns the solution for the sub-problem identified by the component. If the solution is in
    /// the cache, it is not computed. Otherwise it is solved and stored in the cache.
    fn get_cached_component_or_compute(&mut self, component: ComponentIndex, level: isize, bound_factor: f64) -> (Bounds, isize) {
        self.statistics.cache_access();
        let bit_repr = self.graph.get_bit_representation(&self.state, component, &self.component_extractor);
        match self.cache.get(&bit_repr) {
            None => {
                self.statistics.cache_miss();
                let f = self.choose_and_branch(component, level, bound_factor);
                self.cache.insert(bit_repr, f.0.clone());
                f
            },
            Some(f) => {
                if self.are_bounds_tight_enough(f.0.clone(), 1.0 - f.1.clone(), bound_factor) {
                    (f.clone(), level)
                } else {
                    let new_f = self.choose_and_branch(component, level, bound_factor);
                    self.cache.insert(bit_repr, new_f.0.clone());
                    new_f
                }
            }
        }
    }

    fn get_probability_mass_from_assignment(&mut self, distribution: DistributionIndex, target: f64, component: ComponentIndex) -> (Float, Float) {
        let mut added_proba = f128!(1.0);
        let mut removed_proba = f128!(1.0);
        let mut has_removed = false;
        for distribution in self.component_extractor.component_distribution_iter(component) {
            self.distribution_out_vec[distribution.0] = 0.0;
        }
        for v in self.propagator.assignments_iter(&self.state).map(|l| l.to_variable()) {
            if self.graph[v].is_probabilitic() {
                let d = self.graph[v].distribution().unwrap();
                if self.graph[v].value(&self.state).unwrap() {
                    added_proba *= self.graph[v].weight().unwrap();
                } else if distribution != d {
                    has_removed = true;
                    self.distribution_out_vec[d.0] += self.graph[v].weight().unwrap();
                }
            }
        }
        if has_removed {
            for distribution in self.component_extractor.component_distribution_iter(component).filter(|d| *d != distribution) {
                let v = self.distribution_out_vec[distribution.0];
                let remain_d = self.graph[distribution].remaining(&self.state);
                removed_proba *= remain_d - v;
                if v != 0.0 {
                    self.graph[distribution].remove_probability_mass(v, &mut self.state);
                }
            }
        } else {
            removed_proba.assign(target);
        }
        (added_proba, target - removed_proba)
    }

    /// Chooses a distribution to branch on using the heuristics of the solver and returns the
    /// solution of the component.
    /// The solution is the sum of the probability of the SAT children.
    fn choose_and_branch(&mut self, component: ComponentIndex, level: isize, bound_factor: f64)-> (Bounds, isize) {
        let decision = self.branching_heuristic.branch_on(
            &self.graph,
            &mut self.state,
            &self.component_extractor,
            component,
        );
        let target = self.component_extractor[component].max_probability();
        if let Some(distribution) = decision {
            self.statistics.or_node();
            let mut p_in = f128!(0.0);
            let mut p_out = f128!(0.0);
            let mut branches = self.graph[distribution].iter_variables().filter(|v| !self.graph[*v].is_fixed(&self.state)).collect::<Vec<VariableIndex>>();
            branches.sort_by(|v1, v2| {
                let f1 = self.graph[*v1].weight().unwrap();
                let f2 = self.graph[*v2].weight().unwrap();
                f1.total_cmp(&f2)
            });
            let unsat_factor = target / self.graph[distribution].remaining(&self.state);
            for variable in branches.iter().rev().copied() {
                let v_weight = self.graph[variable].weight().unwrap();
                self.state.save_state();
                match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, component, &mut self.component_extractor, level) {
                    Err(backtrack_level) => {
                        self.statistics.unsat();
                        self.branching_heuristic.update_distribution_score(distribution);
                        self.branching_heuristic.decay_scores();
                        self.restore();
                        let mut f = f128!(1.0);
                        for d in self.component_extractor.component_distribution_iter(component) {
                            if d != distribution {
                                f *= self.graph[d].remaining(&self.state);
                            }
                        }
                        p_out += v_weight*unsat_factor;
                        if backtrack_level != level {
                            debug_assert!(p_in == 0.0);
                            self.restore();
                            return ((p_in, f128!(target)), backtrack_level);
                        }
                    },
                    Ok(_) => {
                        let (m_in, m_out) = self.get_probability_mass_from_assignment(distribution, target / self.graph[distribution].remaining(&self.state), component);
                        p_out += v_weight * m_out;
                        if m_in != 0.0 {
                            let (child_sol, backtrack_level) = self._solve(component, level+1, bound_factor);
                            p_in += child_sol.0 * &m_in;
                            p_out += child_sol.1 * &m_in;
                            if backtrack_level != level {
                                self.restore();
                                return ((f128!(0.0), f128!(target)), backtrack_level);
                            }
                            let lb = p_in.clone();
                            let ub = target - p_out.clone();
                            if self.are_bounds_tight_enough(lb, ub, bound_factor) {
                                self.restore();
                                return ((p_in, p_out), level);
                            }
                        }
                        self.restore();
                    }
                };
            }
            ((p_in, p_out), level)
        } else {
            // The sub-formula is SAT, by definition return 1. In practice should not happen since in that case no components are returned in the detection
            ((f128!(1.0), f128!(0.0)), level)
        }
    }

    /// Solves the problem for the sub-graph identified by component.
    pub fn _solve(&mut self, component: ComponentIndex, level: isize, bound_factor: f64) -> (Bounds, isize) {
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
        let mut target = f128!(1.0);
        for d in self.component_extractor.component_distribution_iter(component) {
            if self.graph[d].is_constrained(&self.state) {
                target *= self.graph[d].remaining(&self.state);
            }
        }
        // First we detect the sub-components in the graph
        if self
            .component_extractor
            .detect_components(&mut self.graph, &mut self.state, component, &mut self.propagator)
        {
            self.statistics.and_node();
            let number_component = self.component_extractor.number_components(&self.state);
            let new_bound_factor = bound_factor.powf(1.0 / number_component as f64);
            self.statistics.decomposition(number_component);
            for sub_component in self.component_extractor.components_iter(&self.state) {
                let sub_target = self.component_extractor[sub_component].max_probability();
                let (cache_solution, _) = self.get_cached_component_or_compute(sub_component, level, new_bound_factor);
                p_in *= cache_solution.0;
                p_out *= sub_target - cache_solution.1;
            }
        }
        self.restore();
        ((p_in, target - p_out), level - 1)
    }

    pub fn preproc(&mut self) -> Result<(), Unsat> {
        self.preproc_out.assign(1.0);
        self.preproc_in.assign(0.0);
        let preproc = Preprocessor::new(&mut self.graph, &mut self.state, &mut *self.branching_heuristic, &mut self.propagator, &mut self.component_extractor).preprocess(false);
        if preproc.is_none() {
            return Err(Unsat);
        }
        self.preproc_in = preproc.unwrap();
        
        for distribution in self.graph.distributions_iter() {
            let mut sum_neg = 0.0;
            for variable in self.graph[distribution].iter_variables() {
                if let Some(v) = self.graph[variable].value(&self.state) {
                    if v {
                        sum_neg = 0.0;
                        break;
                    } else {
                        sum_neg += self.graph[variable].weight().unwrap();
                    }
                }
            }
            self.preproc_out *= 1.0 - sum_neg;
        }
        self.preproc_out = 1.0 - self.preproc_out.clone();
        self.branching_heuristic.init(&self.graph, &self.state);
        self.propagator.set_forced();
        Ok(())
    }

    pub fn solve(&mut self) -> ProblemSolution {
        self.state.save_state();
        if let Err(_) = self.preproc() {
            return ProblemSolution::Err(Unsat);
        }
        let (solution, _) = self._solve(ComponentIndex(0), 1, (1.0 + self.epsilon).powf(2.0));
        self.statistics.print();
        let ub: Float = 1.0 - (&self.preproc_out + solution.1.clone()*&self.preproc_in) / &self.prefix_factor;
        let lb: Float = (self.preproc_in.clone() * solution.0) / &self.prefix_factor;
        let proba: Float = (ub*lb).sqrt();
        self.restore();
        ProblemSolution::Ok(proba)
    }
    
    pub fn init(&mut self) {
        self.propagator.init(self.graph.number_clauses());
    }


    #[inline]
    fn are_bounds_tight_enough(&self, lb: Float, ub: Float, bound_factor: f64) -> bool {
        ub <= lb.clone()*bound_factor
    }
}
