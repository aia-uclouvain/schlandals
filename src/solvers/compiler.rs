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


use rustc_hash::FxHashMap;
use search_trail::{StateManager, SaveAndRestore};

use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::*;
use crate::branching::BranchingDecision;
use crate::diagrams::semiring::SemiRing;
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use crate::common::*;
use crate::diagrams::dac::dac::{NodeIndex, Dac};

/// The solver for a particular set of Horn clauses. It is generic over the branching heuristic
/// and has a constant parameter that tells if statistics must be recorded or not.
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
    /// If true, allows the compiler to compile partially the diagram and run an approximate solver
    /// to estimate the probability of the partially compiled node
    partial: bool,
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
    ) -> Self {
        let cache = FxHashMap::default();
        Self {
            graph,
            state,
            component_extractor,
            branching_heuristic,
            propagator,
            cache,
            partial: false,
        }
    }

    pub fn set_partial(&mut self, value: bool) {
        self.partial = value;
    }

    fn restore(&mut self) {
        self.propagator.restore(&self.state);
        self.state.restore_state();
    }

    fn expand_sum_node<R>(&mut self, dac: &mut Dac<R>, component: ComponentIndex, distribution: DistributionIndex, level: isize, number_constrained_distribution: usize) -> Option<NodeIndex> 
        where R: SemiRing
    {

        let mut children: Vec<NodeIndex> = vec![];
        for variable in self.graph[distribution].iter_variables() {
            self.state.save_state();
            match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, component, &mut self.component_extractor, level) {
                Err(_) => { },
                Ok(_) => {
                    if let Some(child) = self.expand_prod_node(dac, component, level + 1, number_constrained_distribution) {
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
    
    fn expand_prod_node<R>(&mut self, dac: &mut Dac<R>, component: ComponentIndex, level: isize, number_constrained_distribution: usize) -> Option<NodeIndex>
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
            let distributions_in_components = self.component_extractor.components_iter(&self.state).map(|c| self.component_extractor.component_number_distribution(c)).sum::<usize>();
            for sub_component in self.component_extractor.components_iter(&self.state) {
                let bit_repr = self.graph.get_bit_representation(&self.state, sub_component, &self.component_extractor);
                if !self.cache.contains_key(&bit_repr) {
                    let distribution_in_branches = number_constrained_distribution - distributions_in_components + self.component_extractor.component_number_distribution(sub_component);
                    let number_distribution_component = self.component_extractor.component_number_distribution(sub_component);
                    let ratio_distrib = (distribution_in_branches - number_distribution_component) as f64 / level as f64;
                    let should_approximate = self.partial && level >= (0.1 * distribution_in_branches as f64) as isize && ratio_distrib <= 2.0;
                    let branching = self.branching_heuristic.branch_on(&self.graph, &mut self.state, &self.component_extractor, sub_component);
                    if !should_approximate && branching.is_some() {
                        let distribution = branching.unwrap();
                        if let Some(child) = self.expand_sum_node(dac, sub_component, distribution, level, distribution_in_branches) {
                            sum_children.push(child);
                            self.cache.insert(bit_repr, Some(child));
                        } else {
                            self.cache.insert(bit_repr, None);
                            prod_node = None;
                            sum_children.clear();
                            break;
                        }
                    } else {
                        if self.component_extractor.component_distribution_iter(sub_component).find(|d| self.graph[*d].is_constrained(&self.state)).is_some() {
                            let child = dac.add_sum_node();
                            for (variable, value) in  self.propagator.iter_propagated_assignments().map(|l| (l.to_variable(), l.is_positive())).filter(|(l, _)| self.graph[*l].is_probabilitic() && self.graph[*l].reason(&self.state).is_none()) {
                                dac[child].add_to_propagation(variable, value);
                            }
                            dac[child].add_distributions(self.graph.number_distributions(), self.component_extractor.component_distribution_iter(sub_component));
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
            for child in sum_children {
                dac.add_node_output(child, node);
            }
        }
        prod_node
    }

    pub fn compile<R>(&mut self) -> Option<Dac<R>>
        where R: SemiRing
    {

        // First set the number of clause in the propagator. This can not be done at the initialization of the propagator
        // because we need it to parse the input file as some variables might be detected as always being true or false.
        /* let mut probabilities: Vec<Vec<f64>>= vec![];
        for distribution in self.graph.distributions_iter() {
            let proba: Vec<f64>= self.graph[distribution].iter_variables().map(|v|self.graph[v].weight().unwrap()).collect();
            probabilities.push(proba);
        } */
        self.propagator.init(self.graph.number_clauses());
        let preproc = Preprocessor::new(&mut self.graph, &mut self.state, &mut *self.branching_heuristic, &mut self.propagator, &mut self.component_extractor).preprocess(false);
        if preproc.is_none() {
            return None;
        }
        let number_constrained_distribution = self.graph.distributions_iter().filter(|d| self.graph[*d].is_constrained(&self.state)).count();
        self.branching_heuristic.init(&self.graph, &self.state);
        let mut dac = Dac::new();
        match self.expand_prod_node(&mut dac, ComponentIndex(0), 1, number_constrained_distribution) {
            None => None,
            Some(_) => Some(dac),
        }
    }

    pub fn extend_partial_node_with<R>(&mut self, node: NodeIndex, dac: &mut Dac<R>, distribution: DistributionIndex)
        where R: SemiRing
    {
        debug_assert!(dac[node].is_sum());
        self.state.save_state();
        let propagations = dac[node].get_propagation().clone();
        for (variable, value) in propagations.iter().copied() {
            self.propagator.add_to_propagation_stack(variable, value, None);
        }
        match self.propagator.propagate(&mut self.graph, &mut self.state, ComponentIndex(0), &mut self.component_extractor, 0) {
            Ok(_) => {
                let mut children: Vec<NodeIndex> = vec![];
                for variable in self.graph[distribution].iter_variables() {
                    self.state.save_state();
                    match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, ComponentIndex(0), &mut self.component_extractor, 1) {
                        Err(_) => { },
                        Ok(_) => {
                            if let Some(child) = self.expand_prod_node(dac, ComponentIndex(0), 2, 0) {
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
                }
            },
            Err(_) => {
                panic!("Trying to extend with UNSAT node");
            },
        }
        self.state.restore_state();
    }
}
