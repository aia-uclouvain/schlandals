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
use crate::core::graph::{DistributionIndex, ClauseIndex, Graph};
use crate::core::literal::Literal;
use crate::branching::BranchingDecision;
use crate::diagrams::semiring::SemiRing;
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use crate::common::*;
use crate::diagrams::dac::dac::{NodeIndex, Dac};

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
        Self {
            graph,
            state,
            component_extractor,
            branching_heuristic,
            propagator,
            cache,
            number_constrained_distribution: 0,
            epsilon
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
        if level == 1 {
            dac.set_number_used_distributions(self.graph.number_distributions());
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
                if level ==1 {
                    for distribution in self.component_extractor.component_distribution_iter(sub_component) {
                    dac.set_used_distribution(distribution);
                    }
                }

                let bit_repr = self.graph.get_bit_representation(&self.state, sub_component, &self.component_extractor);
                if !self.cache.contains_key(&bit_repr) {
                    if let Some(distribution) = self.branching_heuristic.branch_on(&self.graph, &mut self.state, &self.component_extractor, sub_component) {
                        if let Some(child) = self.expand_sum_node(dac, sub_component, distribution, level, new_bound_factor) {
                            sum_children.push(child);
                            self.cache.insert(bit_repr, Some(child));
                        } else {
                            self.cache.insert(bit_repr, None);
                            prod_node = None;
                            sum_children.clear();
                            break;
                        }
                    } else if self.component_extractor.component_iter(sub_component).find(|clause| !self.graph[*clause].is_learned() && self.graph[*clause].has_probabilistic(&self.state)  && self.graph[*clause].is_constrained(&self.state)).is_some() {
                        // TODO : check if this is correct
                        let child = dac.add_sum_node();
                        for (variable, value) in self.propagator.iter_propagated_assignments().map(|l| (l.to_variable(), l.is_positive())).filter(|(l, _)| (self.graph[*l].reason(&self.state).is_none())) {
                            dac[child].add_to_propagation(variable, value);
                        }
                        for clause in self.component_extractor.component_iter(sub_component) {
                            dac[child].add_to_clauses(clause);
                        }
                        dac.add_to_partial_list(child);
                        dac[child].add_distributions(self.graph.number_distributions(), self.component_extractor.component_distribution_iter(sub_component));
                        dac[child].set_bounding_factor(new_bound_factor);
                        sum_children.push(child);
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
                self.tag_unsat_partial_nodes(&mut dac);
                Some(dac)
            }
        };
        self.restore();
        ret
    }
    
    pub fn tag_unsat_partial_nodes<R>(&mut self, dac:&mut Dac<R>)
        where R: SemiRing
    {
        let mut changed = true;
        while changed {
            changed = false;
            // Since the nodes are removed from the vector of partial node, we traverse in reverse
            // order so we do not mess up the indexes (the nodes are removed from the vector with
            // swap_remove)
            for node_i in (0..dac.number_partial_nodes()).rev() {
                let node = dac.get_partial_node_at(node_i);
                if !dac[node].is_unsat() && dac[node].is_node_incomplete(){
                    if self.is_partial_node_unsat(&dac, node){
                        changed = true;
                        dac.set_partial_node_unsat(node);
                    }
                }
            }
        }
    }
    pub fn is_partial_node_unsat<R>(&mut self, dac:&Dac<R>, node: NodeIndex) -> bool 
        where R: SemiRing
    {
        self.state.save_state();
        let propagations = dac[node].get_propagation();
        for (variable, value) in propagations.iter().copied().rev() {
            self.propagator.add_to_propagation_stack(variable, value, None);
        }
        let res = self.propagator.propagate(&mut self.graph, &mut self.state, ComponentIndex(0), &mut self.component_extractor, 0, true).is_err();
        self.restore();
        res
    }

    pub fn get_learned_clauses(&self) -> Vec<Vec<Literal>> {
        self.graph.clauses_iter().filter(|c| self.graph[*c].is_learned()).map(|c| self.get_clause(c)).collect()
    }

    pub fn get_clause(&self, clause: ClauseIndex) -> Vec<Literal> {
        self.graph[clause].iter().collect()
    }
}
