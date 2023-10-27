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
use crate::heuristics::BranchingDecision;
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use super::statistics::Statistics;
use crate::common::*;
use crate::PEAK_ALLOC;

use rug::Float;

use super::*;

type DiscrepencyChoice = (DistributionIndex, usize);

/// The solver for a particular set of Horn clauses. It is generic over the branching heuristic
/// and has a constant parameter that tells if statistics must be recorded or not.
pub struct LDSSolver<'b, B, const S: bool>
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
    propagator: Propagator,
    /// Cache used to store results of sub-problems
    cache: FxHashMap<CacheEntry, (Bounds, DiscrepencyChoice)>,
    /// Statistics collectors
    statistics: Statistics<S>,
    /// Memory limit allowed for the solver. This is a global memory limit, not a cache-size limit
    mlimit: u64,
    /// Vector used to store probability mass out of the distribution in each node
    distribution_out_vec: Vec<f64>,
    epsilon: f64,
    backtrack_level: isize,
    nodes_explored: usize,
}

impl<'b, B, const S: bool> LDSSolver<'b, B, S>
where
    B: BranchingDecision + ?Sized,
{
    pub fn new(
        graph: Graph,
        state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: &'b mut B,
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
            backtrack_level: 0,
            nodes_explored: 0,
        }
    }
    
    fn restore(&mut self) {
        self.propagator.restore(&self.state);
        self.state.restore_state();
    }

    /// Returns the solution for the sub-problem identified by the component. If the solution is in
    /// the cache, it is not computed. Otherwise it is solved and stored in the cache.
    fn get_cached_component_or_compute(&mut self, component: ComponentIndex, level: isize, bound_factor: f64, discrepency: usize) -> (Bounds, usize) {
        self.statistics.cache_access();
        let bit_repr = self.graph.get_bit_representation(&self.state, component, &self.component_extractor);
        match self.cache.get(&bit_repr) {
            None => {
                self.statistics.cache_miss();
                self.nodes_explored += 1;
                let (f, d) = self.choose_and_branch(component, level, bound_factor, discrepency, None);
                self.cache.insert(bit_repr, (f.clone(), d));
                (f, d.1)
            },
            Some((f, d)) => {
                if d.1 < discrepency {
                    let (new_f, new_d) = self.choose_and_branch(component, level, bound_factor, discrepency, None);
                    self.cache.insert(bit_repr, (new_f.clone(), new_d));
                    (new_f, new_d.1)
                } else {
                    (f.clone(), d.1)
                }
            }
        }
    }
    
    fn probability_mass_from_assignments(&mut self, branched_distribution: DistributionIndex) -> (Float, Float) {
        let mut p_in = f128!(1.0);
        let mut p_out = f128!(1.0);
        let mut has_removed = false;
        self.distribution_out_vec.fill(0.0);
        for v in self.propagator.assignments_iter(&self.state).map(|l| l.to_variable()) {
            if self.graph[v].is_probabilitic() {
                let d = self.graph[v].distribution().unwrap();
                if self.graph[v].value(&self.state).unwrap() {
                    p_in *= self.graph[v].weight().unwrap();
                } else if branched_distribution != d {
                    has_removed = true;
                    self.distribution_out_vec[d.0] += self.graph[v].weight().unwrap();
                }
            }
        }
        if has_removed {
            for v in self.distribution_out_vec.iter().copied() {
                p_out *= 1.0 - v;
            }
        }
        (p_in, p_out)
    }

    /// Chooses a distribution to branch on using the heuristics of the solver and returns the
    /// solution of the component.
    /// The solution is the sum of the probability of the SAT children.
    fn choose_and_branch(&mut self,
        component: ComponentIndex,
        level: isize,
        bound_factor: f64,
        discrepency: usize,
        forced_choice: Option<DistributionIndex>) -> (Bounds, DiscrepencyChoice) {
        let decision = match forced_choice {
            None => {
                self.branching_heuristic.branch_on(
                    &self.graph,
                    &self.state,
                    &self.component_extractor,
                    component,
                    )
            },
            Some(d) => Some(d),
        };
        if let Some(distribution) = decision {
            self.statistics.or_node();
            let mut p_in = f128!(0.0);
            let mut p_out = f128!(0.0);
            let mut branches = self.graph[distribution].iter_variables().collect::<Vec<VariableIndex>>();
            branches.sort_by(|v1, v2| {
                let f1 = self.graph[*v1].weight().unwrap();
                let f2 = self.graph[*v2].weight().unwrap();
                f1.total_cmp(&f2)
            });
            let mut visited = 0;
            let mut node_discrepency = usize::MAX;
            for (branch_idx, variable) in branches.iter().copied().enumerate() {
                if branch_idx == discrepency {
                    break;
                }
                if self.graph[variable].is_fixed(&self.state) {
                    continue;
                }
                let v_weight = self.graph[variable].weight().unwrap();
                self.state.save_state();
                match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, component, &mut self.component_extractor, level) {
                    Err(backtrack_level) => {
                        self.statistics.unsat();
                        if backtrack_level != level {
                            debug_assert!(p_in == 0.0);
                            self.restore();
                            self.backtrack_level = backtrack_level;
                            return ((f128!(0.0), f128!(1.0)), (distribution, usize::MAX)); 
                        }
                        p_out += v_weight;
                    },
                    Ok(_) => {
                        let (in_mass, out_mass) = self.probability_mass_from_assignments(distribution);
                        p_out += v_weight * (1.0 - out_mass);
                        let new_discrepency = discrepency - branch_idx;
                        let (child_sol, child_discrepency) = self._solve(component, level+1, bound_factor, new_discrepency);
                        if self.backtrack_level != level {
                            println!("Here?");
                            self.restore();
                            return ((f128!(0.0), f128!(1.0)), (distribution, usize::MAX));
                        }
                        p_in += child_sol.0 * &in_mass;
                        p_out += child_sol.1 * &in_mass;
                        node_discrepency = node_discrepency.min(child_discrepency);
                    }
                };
                visited += 1;
                self.restore();
            }
            if visited != branches.len() {
                node_discrepency = node_discrepency.min(visited);
            }
            self.backtrack_level = level;
            ((p_in, p_out), (distribution, node_discrepency))
        } else {
            debug_assert!(false);
            // The sub-formula is SAT, by definition return 1. In practice should not happen since in that case no components are returned in the detection
            self.backtrack_level = level;
            ((f128!(1.0), f128!(0.0)), (DistributionIndex(0), usize::MAX))
        }
    }

    /// Solves the problem for the sub-graph identified by component.
    pub fn _solve(&mut self, component: ComponentIndex, level: isize, bound_factor: f64, discrepency: usize) -> (Bounds, usize) {
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
        // First we detect the sub-components in the graph
        let mut children_discrepency = usize::MAX;
        if self
            .component_extractor
            .detect_components(&mut self.graph, &mut self.state, component, &mut self.propagator)
        {
            self.statistics.and_node();
            let number_component = self.component_extractor.number_components(&self.state);
            let new_bound_factor = bound_factor.powf(1.0 / number_component as f64);
            self.statistics.decomposition(number_component);
            self.backtrack_level = level;
            for sub_component in self.component_extractor.components_iter(&self.state) {
                let (cache_solution, node_discrepency) = self.get_cached_component_or_compute(sub_component, level, new_bound_factor, discrepency);
                if self.backtrack_level != level {
                    self.state.restore_state();
                    return ((f128!(0.0), f128!(1.0)), usize::MAX);
                }
                children_discrepency = children_discrepency.min(node_discrepency);
                p_in *= &cache_solution.0;
                p_out *= 1.0 - cache_solution.1;
                if p_in == 0.0 {
                    break;
                }
            }
        }
        self.state.restore_state();
        self.backtrack_level = level - 1;
        ((p_in, 1.0 - p_out), children_discrepency)
    }
    
    pub fn solve(&mut self) -> ProblemSolution {
        self.propagator.init(self.graph.number_clauses());
        let preproc = Preprocessor::new(&mut self.graph, &mut self.state, self.branching_heuristic, &mut self.propagator, &mut self.component_extractor).preprocess(false);
        if preproc.is_none() {
            return ProblemSolution::Err(Unsat);
        }
        let mut p_in = preproc.unwrap();
        let mut p_out = f128!(1.0);
        
        for distribution in self.graph.distributions_iter() {
            let mut sum_neg = 0.0;
            for variable in self.graph[distribution].iter_variables() {
                if let Some(v) = self.graph[variable].value(&self.state) {
                    if v {
                        p_in *= self.graph[variable].weight().unwrap();
                        sum_neg = 0.0;
                        break;
                    } else {
                        sum_neg += self.graph[variable].weight().unwrap();
                    }
                }
            }
            p_out *= 1.0 - sum_neg;
        }
        p_out = 1.0 - p_out;
        self.branching_heuristic.init(&self.graph, &self.state);
        self.propagator.set_forced();
        
        let mut proba: Option<Float> = None;
        for discrepency in 1..11 {
            let (solution, _) = self._solve(ComponentIndex(0), 1, (1.0 + self.epsilon).sqrt(), discrepency);
            let ub: Float = 1.0 - solution.1*(1.0 - p_out.clone());
            println!("{} {} {} {}", discrepency, solution.0, ub, self.nodes_explored);
            proba = Some(p_in.clone() * (solution.0 * &ub).sqrt());
            
        }
        self.statistics.print();
        ProblemSolution::Ok(proba.unwrap())
    }
    
    #[inline]
    fn are_bounds_tight_enough(&self, lb: Float, ub: Float, bound_factor: f64) -> bool {
        ub <= lb.clone()*bound_factor
    }
}