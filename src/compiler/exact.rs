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
use search_trail::{StateManager, SaveAndRestore};

use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::*;
use crate::heuristics::BranchingDecision;
use crate::propagator::CompiledPropagator;
use crate::common::*;
use crate::compiler::circuit::*;

/// The solver for a particular set of Horn clauses. It is generic over the branching heuristic
/// and has a constant parameter that tells if statistics must be recorded or not.
pub struct ExactDACCompiler<'b, B>
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
    propagator: CompiledPropagator,
    /// Cache used to store results of sub-problems
    cache: FxHashMap<CacheEntry, Option<CircuitNodeIndex>>,
}

impl<'b, B> ExactDACCompiler<'b, B>
where
    B: BranchingDecision + ?Sized,
{
    pub fn new(
        graph: Graph,
        state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: &'b mut B,
        propagator: CompiledPropagator,
    ) -> Self {
        let cache = FxHashMap::default();
        Self {
            graph,
            state,
            component_extractor,
            branching_heuristic,
            propagator,
            cache,
        }
    }

    fn expand_sum_node(&mut self, dac: &mut Dac, component: ComponentIndex, distribution: DistributionIndex) -> Option<CircuitNodeIndex> {
        let mut children: Vec<CircuitNodeIndex> = vec![];
        for variable in self.graph.distribution_variable_iter(distribution) {
            self.state.save_state();
            match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, component, &self.component_extractor) {
                Err(_) => { },
                Ok(_) => {
                    if let Some(child) = self.expand_prod_node(dac, component) {
                        children.push(child);
                    }
                }
            }
            self.state.restore_state();
        }
        if !children.is_empty() {
            let node = dac.add_sum_node();
            for child in children {
                dac.add_circuit_node_output(child, node);
            }
            Some(node)
        } else {
            None
        }
    }
    
    fn expand_prod_node(&mut self, dac: &mut Dac, component: ComponentIndex) -> Option<CircuitNodeIndex> {
        let mut prod_node: Option<CircuitNodeIndex> = if self.propagator.has_assignments() || self.propagator.has_unconstrained_distribution() {
            let node = dac.add_prod_node();
            for (distribution, variable, value) in self.propagator.assignments_iter() {
                if value {
                    let value_id = variable.0 - self.graph.get_distribution_start(distribution).0;
                    dac.add_distribution_output(distribution, node, value_id);
                }
            }
        
            for distribution in self.propagator.unconstrained_distributions_iter() {
                if self.graph.distribution_number_false(distribution, &self.state) != 0 {
                    let sum_node = dac.add_sum_node();
                    for variable in self.graph.distribution_variable_iter(distribution) {
                        if !self.graph.is_variable_fixed(variable, &self.state) {
                            let value_id = variable.0 - self.graph.get_distribution_start(distribution).0;
                            dac.add_distribution_output(distribution, sum_node, value_id);
                        }
                    }
                    dac.add_circuit_node_output(sum_node, node);
                }
            }
            Some(node)
        } else {
            None
        };
        let mut sum_children: Vec<CircuitNodeIndex> = vec![];
        if self.component_extractor.detect_components(&mut self.graph, &mut self.state, component, &mut self.propagator) {
            for sub_component in self.component_extractor.components_iter(&self.state) {
                let bit_repr = self.graph.get_bit_representation(&self.state, sub_component, &self.component_extractor);
                match self.cache.get(&bit_repr) {
                    None => {
                        if let Some(distribution) = self.branching_heuristic.branch_on(&self.graph, &self.state, &self.component_extractor, sub_component) {
                            if let Some(child) = self.expand_sum_node(dac, sub_component, distribution) {
                                sum_children.push(child);
                                self.cache.insert(bit_repr, Some(child));
                            } else {
                                self.cache.insert(bit_repr, None);
                                prod_node = None;
                                sum_children.clear();
                                break;
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
                dac.add_circuit_node_output(child, node);
            }
        }
        prod_node
    }

    pub fn compile(&mut self) -> Option<Dac> {
        // First set the number of clause in the propagator. This can not be done at the initialization of the propagator
        // because we need it to parse the input file as some variables might be detected as always being true or false.
        self.propagator.set_number_clauses(self.graph.number_clauses());
        // Doing an initial propagation to detect some UNSAT formula from the start
        match self.propagator.propagate(&mut self.graph, &mut self.state, ComponentIndex(0), &self.component_extractor) {
            Err(_) => None,
            Ok(_) => {
                self.branching_heuristic.init(&self.graph, &self.state);
                let mut dac = Dac::new(&self.graph);
                match self.expand_prod_node(&mut dac, ComponentIndex(0)) {
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
    }
}