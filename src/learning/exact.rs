//Schlandal
//Copyright (C) 2022 A. Dubray
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
use search_trail::{StateManager, SaveAndRestore, ReversibleUsize, UsizeManager};

use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::*;
use crate::branching::BranchingDecision;
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use crate::common::*;
use crate::learning::circuit::*;
use std::time::SystemTime;

/// The solver for a particular set of Horn clauses. It is generic over the branching heuristic
/// and has a constant parameter that tells if statistics must be recorded or not.
pub struct DACCompiler<'b, B>
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
    cache: FxHashMap<CacheEntry, Option<NodeIndex>>,
    /// limit on the number of distributions in all branches
    limit: usize,
    /// Current number of distributions in the branch
    distribution_count: ReversibleUsize,
}

impl<'b, B> DACCompiler<'b, B>
where
    B: BranchingDecision + ?Sized,
{
    pub fn new(
        graph: Graph,
        mut state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: &'b mut B,
        propagator: Propagator,
        limit: usize,
    ) -> Self {
        let cache = FxHashMap::default();
        let distribution_count = state.manage_usize(0);
        Self {
            graph,
            state,
            component_extractor,
            branching_heuristic,
            propagator,
            cache,
            limit,
            distribution_count,
        }
    }

    fn expand_sum_node(&mut self, dac: &mut Dac, component: ComponentIndex, distribution: DistributionIndex, level: isize, start:SystemTime, timeout:u64) -> Option<NodeIndex> {
        if start.elapsed().unwrap().as_secs() > timeout {
            return None;
        }

        let mut children: Vec<NodeIndex> = vec![];
        for variable in self.graph[distribution].iter_variables() {
            self.state.save_state();
            match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, component, &mut self.component_extractor, level) {
                Err(_) => { },
                Ok(_) => {
                    if let Some(child) = self.expand_prod_node(dac, component, level + 1, start, timeout) {
                        children.push(child);
                    }
                    if start.elapsed().unwrap().as_secs() > timeout {
                        return None;
                    }
                }
            }
            self.state.restore_state();
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
    
    fn expand_prod_node(&mut self, dac: &mut Dac, component: ComponentIndex, level: isize, start:SystemTime, timeout:u64) -> Option<NodeIndex> {
        if start.elapsed().unwrap().as_secs() > timeout {
            return None;
        }
        
        let mut prod_node: Option<NodeIndex> = if self.propagator.has_assignments() || self.propagator.has_unconstrained_distribution() {
            let node = dac.add_prod_node();
            for literal in self.propagator.assignments_iter(&self.state) {
                let variable = literal.to_variable();
                if self.graph[variable].is_probabilitic() && self.graph[variable].value(&self.state).unwrap() {
                    let distribution = self.graph[variable].distribution().unwrap();
                    let value_id = variable.0 - self.graph[distribution].start().0;
                    let disti_index = dac.get_distribution_value_node_index(distribution, value_id, self.graph[variable].weight().unwrap());
                    dac.add_node_output(disti_index, node);
                    self.state.increment_usize(self.distribution_count);
                }
            }
        
            for distribution in self.propagator.unconstrained_distributions_iter() {
                if self.graph[distribution].number_false(&self.state) != 0 {
                    let sum_node = dac.add_sum_node();
                    self.state.increment_usize(self.distribution_count);
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
            for sub_component in self.component_extractor.components_iter(&self.state) {
                let bit_repr = self.graph.get_bit_representation(&self.state, sub_component, &self.component_extractor);
                match self.cache.get(&bit_repr) {
                    None => {
                        if self.state.get_usize(self.distribution_count) < self.limit {
                            if let Some(distribution) = self.branching_heuristic.branch_on(&self.graph, &mut self.state, &self.component_extractor, sub_component) {
                                self.state.increment_usize(self.distribution_count);
                                if let Some(child) = self.expand_sum_node(dac, sub_component, distribution, level, start, timeout) {
                                    sum_children.push(child);
                                    self.cache.insert(bit_repr, Some(child));
                                } else {
                                    self.cache.insert(bit_repr, None);
                                    prod_node = None;
                                    sum_children.clear();
                                    break;
                                }
                                if start.elapsed().unwrap().as_secs() > timeout {
                                    return None;
                                }
                            }
                        } else {
                            if self.component_extractor.component_distribution_iter(sub_component).find(|d| self.graph[*d].is_constrained(&self.state)).is_some() {
                                let child = dac.add_sum_node();
                                let propagated = self.propagator.iter_propagated_assignments().map(|l| (l.to_variable(), l.is_positive())).collect::<Vec<(VariableIndex, bool)>>();
                                dac.set_node_propagations(child, propagated);
                                sum_children.push(child);
                            }
                        }
                    },
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

    pub fn compile(&mut self, timeout:u64) -> Option<Dac> {
        let start = SystemTime::now();

        // First set the number of clause in the propagator. This can not be done at the initialization of the propagator
        // because we need it to parse the input file as some variables might be detected as always being true or false.
        /* let mut probabilities: Vec<Vec<f64>>= vec![];
        for distribution in self.graph.distributions_iter() {
            let proba: Vec<f64>= self.graph[distribution].iter_variables().map(|v|self.graph[v].weight().unwrap()).collect();
            probabilities.push(proba);
        } */
        self.propagator.init(self.graph.number_clauses());
        let preproc = Preprocessor::new(&mut self.graph, &mut self.state, self.branching_heuristic, &mut self.propagator, &mut self.component_extractor).preprocess(false);
        if preproc.is_none() {
            return None;
        }
        self.branching_heuristic.init(&self.graph, &self.state);
        let mut dac = Dac::new();
        match self.expand_prod_node(&mut dac, ComponentIndex(0), 1, start, timeout) {
            None => None,
            Some(_) => {
                dac.remove_dead_ends();
                dac.reduce();
                dac.layerize();
                Some(dac)
            }
        }
    }
}
