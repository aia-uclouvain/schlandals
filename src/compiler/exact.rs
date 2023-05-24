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
use crate::solver::branching::BranchingDecision;
use crate::solver::propagator::FTReachablePropagator;
use crate::common::*;
use crate::compiler::aomdd::*;
use rug::Float;

/// The solver for a particular set of Horn clauses. It is generic over the branching heuristic
/// and has a constant parameter that tells if statistics must be recorded or not.
pub struct ExactAOMDDCompiler<'b, B>
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
    propagator: FTReachablePropagator,
    /// Cache used to store results of sub-problems
    cache: FxHashMap<CacheEntry, OrNodeIndex>,
}

impl<'b, B> ExactAOMDDCompiler<'b, B>
where
    B: BranchingDecision + ?Sized,
{
    pub fn new(
        graph: Graph,
        state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: &'b mut B,
        propagator: FTReachablePropagator,
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

    fn expand_or_node(&mut self, aomdd: &mut AOMDD, node: OrNodeIndex, component: ComponentIndex) {
        let distribution = aomdd.get_or_node_decision(node);
        for variable in self.graph.distribution_variable_iter(distribution) {
            self.state.save_state();
            match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, component, &self.component_extractor) {
                Err(_) => {
                    let child_and_node = aomdd.add_and_node(variable, node, f128!(0.0));
                    aomdd.add_and_child(child_and_node, aomdd.get_terminal_inconsistent());
                }
                Ok(v) => {
                    if v != 0.0 {
                        let child_and_node = aomdd.add_and_node(variable, node, v);
                        self.expand_and_node(aomdd, child_and_node, component);
                    } else {
                        let child_and_node = aomdd.add_and_node(variable, node, v);
                        aomdd.add_and_child(child_and_node, aomdd.get_terminal_inconsistent());
                    }
                }
            }
            self.state.restore_state();
        }
    }
    
    fn expand_and_node(&mut self, aomdd: &mut AOMDD, node: AndNodeIndex, component: ComponentIndex) {
        if self
            .component_extractor
            .detect_components(&mut self.graph, &mut self.state, component, &mut self.propagator)
        {
            for sub_component in self.component_extractor.components_iter(&self.state) {
                let bit_repr = self.graph.get_bit_representation(&self.state, sub_component, &self.component_extractor);
                match self.cache.get(&bit_repr) {
                    None => {
                        let decision = self.branching_heuristic.branch_on(
                            &self.graph,
                            &self.state,
                            &self.component_extractor,
                            sub_component,
                        );
                        if let Some(distribution) = decision {
                            let or_node = aomdd.add_or_node(distribution);
                            self.cache.insert(bit_repr, or_node);
                            aomdd.add_and_child(node, or_node);
                            self.expand_or_node(aomdd, or_node, sub_component);
                        } else {
                            self.cache.insert(bit_repr, aomdd.get_terminal_consistent());
                        }
                    },
                    Some(f) => {
                        aomdd.add_and_child(node, *f);
                    }
                }
            }
        } else {
            aomdd.add_and_child(node, aomdd.get_terminal_consistent());
        }
        
    }

    pub fn compile(&mut self) -> AOMDD {
        // First set the number of clause in the propagator. This can not be done at the initialization of the propagator
        // because we need it to parse the input file as some variables might be detected as always being true or false.
        self.propagator.set_number_clauses(self.graph.number_clauses());
        // Doing an initial propagation to detect some UNSAT formula from the start
        let mut aomdd = AOMDD::new();
        match self.propagator.propagate(&mut self.graph, &mut self.state, ComponentIndex(0), &self.component_extractor) {
            Err(_) => aomdd.set_inconsistent_root(),
            Ok(p) => {
                aomdd.set_weight_factor(p);
                // Checks if there are still constrained clauses in the graph
                let mut has_constrained = false;
                for clause in self.graph.clause_iter() {
                    if self.graph.is_clause_constrained(clause, &self.state) {
                        has_constrained = true;
                        break;
                    }
                }
                // If the graph still has constrained clauses, start the search.
                if has_constrained {
                    self.branching_heuristic.init(&self.graph, &self.state);
                    let decision = self.branching_heuristic.branch_on(&self.graph, &self.state, &self.component_extractor, ComponentIndex(0));
                    if let Some(distribution) = decision {
                        let root = aomdd.add_or_node(distribution);
                        aomdd.set_root(root);
                        self.expand_or_node(&mut aomdd, root, ComponentIndex(0));
                    } else {
                        aomdd.set_consistent_root();
                    }
                } else {
                    aomdd.set_consistent_root();
                }
            }
        }
        aomdd
    }
}