//Schlandals
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

use search_trail::{StateManager, SaveAndRestore};

use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::*;
use crate::heuristics::BranchingDecision;
use crate::propagator::Propagator;
use rug::Float;
use crate::common::f128;

use crate::core::variable::Reason;

pub struct Preprocessor<'b, B>
where
    B: BranchingDecision + ?Sized,
{
    /// Implication graph of the input CNF formula
    graph: &'b mut Graph,
    /// State manager that allows to retrieve previous values when backtracking in the search tree
    state: &'b mut StateManager,
    /// Heuristics that decide on which distribution to branch next
    branching_heuristic: &'b mut B,
    /// The propagator
    propagator: &'b mut Propagator,
    /// component extractor
    component_extractor: &'b mut ComponentExtractor,
}

impl<'b, B> Preprocessor<'b, B>
where
    B: BranchingDecision + ?Sized,
{

    pub fn new(graph: &'b mut Graph, state: &'b mut StateManager, branching_heuristic: &'b mut B, propagator: &'b mut Propagator, component_extractor: &'b mut ComponentExtractor) -> Self {
        Self {
            graph,
            state,
            branching_heuristic,
            propagator,
            component_extractor,
        }
    }
    
    pub fn preprocess(&mut self, backbone: bool) -> Option<Float> {
        let mut p = f128!(1.0);

        for variable in self.graph.variables_iter() {
            if self.graph[variable].is_probabilitic() && self.graph[variable].weight().unwrap() == 1.0 {
                self.propagator.add_to_propagation_stack(variable, true, None);
            }
        }
        
        // Find unit clauses
        for clause in self.component_extractor.component_iter(ComponentIndex(0)) {
            if self.graph[clause].is_unit(self.state) {
                let l = self.graph[clause].get_unit_assigment(self.state);
                self.propagator.add_to_propagation_stack(l.to_variable(), l.is_positive(), Some(Reason::Clause(clause)));
            }
        }
        match self.propagator.propagate(self.graph, self.state, ComponentIndex(0), self.component_extractor, 0) {
            Err(_) => return None,
            Ok(_) => {
                p *= self.propagator.get_propagation_prob();
            }
        };
        if backbone {
            let backbone = self.identify_backbone();
            for (variable, value) in backbone.iter().copied() {
                self.propagator.add_to_propagation_stack(variable, value, None);
            }
            if !backbone.is_empty() {
                self.propagator.propagate(&mut self.graph, &mut self.state, ComponentIndex(0), &mut self.component_extractor, 0).unwrap();
                p *= self.propagator.get_propagation_prob().clone();
            }
        }
        Some(p)
    }

    fn identify_backbone(&mut self) -> Vec<(VariableIndex, bool)> {
        let mut backbone: Vec<(VariableIndex, bool)> = vec![];
        for variable in (0..self.graph.number_variables()).map(VariableIndex) {
            if !self.graph[variable].is_probabilitic() && !self.graph[variable].is_fixed(&self.state) {
                self.state.save_state();
                for (v, value) in backbone.iter().copied() {
                    self.propagator.add_to_propagation_stack(v, value, None)
                }
                self.propagator.add_to_propagation_stack(variable, true, None);
                let sat_true = match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, ComponentIndex(0), &mut self.component_extractor, 0) {
                    Err(_) => false,
                    Ok(_) => {
                        self.sat()
                    },
                };
                self.state.restore_state();

                self.state.save_state();
                for (v, value) in backbone.iter().copied() {
                    self.propagator.add_to_propagation_stack(v, value, None)
                }
                let sat_false = match self.propagator.propagate_variable(variable, false, &mut self.graph, &mut self.state, ComponentIndex(0), &mut self.component_extractor, 0) {
                    Err(_) => false,
                    Ok(_) => {
                        self.sat()
                    },
                };
                self.state.restore_state();
                if !sat_true {
                    backbone.push((variable, false));
                }
                if !sat_false {
                    backbone.push((variable, true));
                }
            }
        }
        backbone
    }
    
    fn sat(&mut self) -> bool {
        let decision = self.branching_heuristic.branch_on(
            &self.graph,
            &self.state,
            &self.component_extractor,
            ComponentIndex(0),
        );
        if let Some(distribution) = decision {
            for variable in self.graph[distribution].iter_variables() {
                self.state.save_state();
                match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, ComponentIndex(0), &mut self.component_extractor, 0) {
                    Err(_) => {
                    }
                    Ok(_) => {
                        if self.sat() {
                            self.state.restore_state();
                            return true;
                        }
                    }
                };
                self.state.restore_state();
            }
            false
        } else {
            true
        }
    }
}
