//Schlandal
//Copyright (C) 2022 A. Dubray, L. Dierckx
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


use rustc_hash::{FxHashMap, FxHashSet};
use search_trail::{StateManager, SaveAndRestore};

use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::{DistributionIndex, Graph, VariableIndex};
use crate::branching::BranchingDecision;
use crate::diagrams::semiring::SemiRing;
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use crate::common::*;
use crate::diagrams::dac::dac::{NodeIndex, Dac};
use super::Bounds;
use rug::{Assign, Float};

pub struct DACCompiler<B>
where
    B: BranchingDecision,
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
    cache: FxHashMap<CacheEntry, Option<NodeIndex>>,
    number_constrained_distribution: usize,
    epsilon: f64,
    cache_partial: FxHashMap<CacheEntry, Bounds>,
    distribution_out_vec: Vec<f64>,
}

impl<B> DACCompiler<B>
where
    B: BranchingDecision,
{
    pub fn new(
        graph: Graph,
        state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: Box<B>,
        propagator: Propagator,
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
            number_constrained_distribution: 0,
            epsilon,
            cache_partial: FxHashMap::default(),
            distribution_out_vec,
        }
    }

    fn restore(&mut self) {
        self.propagator.restore(&self.state);
        self.state.restore_state();
    }

    fn expand_sum_node<R>(&mut self, dac: &mut Dac<R>, component: ComponentIndex, distribution: DistributionIndex, level: isize, bounding_factor: f64) -> Option<NodeIndex> 
        where R: SemiRing
    {

        let mut children: Vec<NodeIndex> = vec![];
        for variable in self.graph[distribution].iter_variables() {
            self.state.save_state();
            match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, component, &mut self.component_extractor, level) {
                Err(_) => { },
                Ok(_) => {
                    if let Some(child) = self.expand_prod_node(dac, component, level + 1, bounding_factor) {
                        children.push(child);
                    }
                }
            }
            self.restore();
        }
        if !children.is_empty() {
            let node = dac.add_sum_node();
            for child in children {
                dac.add_node_output(child, node);
            }
            Some(node)
        } else {
            None
        }
    }
    
    fn expand_prod_node<R>(&mut self, dac: &mut Dac<R>, component: ComponentIndex, level: isize, bound_factor: f64) -> Option<NodeIndex>
        where R: SemiRing
    {   
        let mut prod_node: Option<NodeIndex> = if self.propagator.has_assignments() || self.propagator.has_unconstrained_distribution() {
            let node = dac.add_prod_node();
            for literal in self.propagator.assignments_iter(&self.state) {
                let variable = literal.to_variable();
                if self.graph[variable].is_probabilitic() && self.graph[variable].value(&self.state).unwrap() {
                    let distribution = self.graph[variable].distribution().unwrap();
                    let value_id = variable.0 - self.graph[distribution].start().0;
                    let disti_index = dac.get_distribution_value_node_index(distribution, value_id, self.graph[variable].weight().unwrap());
                    dac.add_node_output(disti_index, node);
                }
            }
        
            for distribution in self.propagator.unconstrained_distributions_iter() {
                if self.graph[distribution].number_false(&self.state) != 0 {
                    let sum_node = dac.add_sum_node();
                    for variable in self.graph[distribution].iter_variables() {
                        if !self.graph[variable].is_fixed(&self.state) {
                            let value_id = variable.0 - self.graph[distribution].start().0;
                            let distri_index = dac.get_distribution_value_node_index(distribution, value_id, self.graph[variable].weight().unwrap());
                            dac.add_node_output(distri_index, sum_node);
                        }
                    }
                    dac.add_node_output(sum_node, node);
                }
            }
            Some(node)
        } else {
            None
        };

        let mut sum_children: Vec<NodeIndex> = vec![];
        if self.component_extractor.detect_components(&mut self.graph, &mut self.state, component, &mut self.propagator) {
            let number_component = self.component_extractor.number_components(&self.state);
            let new_bound_factor = bound_factor.powf(1.0 / number_component as f64);
            for sub_component in self.component_extractor.components_iter(&self.state) {

                let bit_repr = self.graph.get_bit_representation(&self.state, sub_component, &self.component_extractor);
                if !self.cache.contains_key(&bit_repr) {
                    if let Some(distribution) = self.branching_heuristic.branch_on(&self.graph, &mut self.state, &self.component_extractor, sub_component) {
                        if self.component_extractor[sub_component].has_learned_distribution() {
                            if let Some(child) = self.expand_sum_node(dac, sub_component, distribution, level, new_bound_factor) {
                                sum_children.push(child);
                                self.cache.insert(bit_repr, Some(child));
                            } else {
                                self.cache.insert(bit_repr, None);
                                prod_node = None;
                                sum_children.clear();
                                break;
                            }
                        } else {
                            let sub_target = self.component_extractor[sub_component].max_probability();
                            let child_bound = self.get_cached_component_or_compute(sub_component, level, new_bound_factor).0;
                            let child_value = (child_bound.0.clone() * (sub_target-child_bound.1.clone())).sqrt().to_f64();
                            let child = dac.add_partial_node(child_value);
                            sum_children.push(child);
                        }
                    }
                } else {
                    match self.cache.get(&bit_repr) {
                        None => {},
                        Some(child) => {
                            match child {
                                None => {
                                    if prod_node.is_some() {
                                        prod_node = None;
                                        sum_children.clear();
                                        break;
                                    }
                                },
                                Some(c) => {
                                    sum_children.push(*c);
                                }
                            };
                        }
                    }
                }
            }
        }
        if !sum_children.is_empty() && prod_node.is_none() {
            prod_node = Some(dac.add_prod_node());
        }
        if let Some(node) = prod_node {
            if level==1 {
                dac.set_root(node);
            }
            for child in sum_children {
                dac.add_node_output(child, node);
            }
        }
        prod_node
    }

    pub fn compile<R>(&mut self) -> Option<Dac<R>>
        where R: SemiRing
    {
        self.state.save_state();
        // First set the number of clause in the propagator. This can not be done at the initialization of the propagator
        // because we need it to parse the input file as some variables might be detected as always being true or false.
        self.propagator.init(self.graph.number_clauses());
        let preproc = Preprocessor::new(&mut self.graph, &mut self.state, &mut *self.branching_heuristic, &mut self.propagator, &mut self.component_extractor).preprocess(false);
        if preproc.is_none() {
            return None;
        }
        let mut unfixed_set: FxHashSet<DistributionIndex> = FxHashSet::default();
        for clause in self.graph.clauses_iter() {
            if self.graph[clause].is_constrained(&self.state){
                for variable in self.graph[clause].iter_probabilistic_variables() {
                    if !self.graph[variable].is_fixed(&self.state) {
                        unfixed_set.insert(self.graph[variable].distribution().unwrap());
                    }
                }
            }
        }
        self.number_constrained_distribution = unfixed_set.len();
        self.branching_heuristic.init(&self.graph, &self.state);
        let mut dac = Dac::new();
        let ret = match self.expand_prod_node(&mut dac, ComponentIndex(0), 1, (1.0 + self.epsilon).powf(2.0)) {
            None => None,
            Some(_) => {
                dac.optimize_structure();
                Some(dac)
            }
        };
        self.restore();
        ret
    }

    // ---- PARTIAL NODE COMPUTATION ----

    pub fn _solve(&mut self, component: ComponentIndex, level: isize, bound_factor: f64) -> (Bounds, isize) {
        // If the memory limit is reached, clear the cache.
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
            let number_component = self.component_extractor.number_components(&self.state);
            let new_bound_factor = bound_factor.powf(1.0 / number_component as f64);
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

    fn get_cached_component_or_compute(&mut self, component: ComponentIndex, level: isize, bound_factor: f64) -> (Bounds, isize) {
        let bit_repr = self.graph.get_bit_representation(&self.state, component, &self.component_extractor);
        match self.cache_partial.get(&bit_repr) {
            None => {
                let f = self.choose_and_branch(component, level, bound_factor);
                self.cache_partial.insert(bit_repr, f.0.clone());
                f
            },
            Some(f) => {
                if self.are_bounds_tight_enough(f.0.clone(), 1.0 - f.1.clone(), bound_factor) {
                    (f.clone(), level)
                } else {
                    let new_f = self.choose_and_branch(component, level, bound_factor);
                    self.cache_partial.insert(bit_repr, new_f.0.clone());
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

    #[inline]
    fn are_bounds_tight_enough(&self, lb: Float, ub: Float, bound_factor: f64) -> bool {
        ub <= lb.clone()*bound_factor
    }

}
