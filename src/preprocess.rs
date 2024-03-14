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

use search_trail::StateManager;

use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::*;
use crate::propagator::Propagator;
use rug::Float;
use crate::common::f128;

pub struct Preprocessor<'b>
{
    /// Implication graph of the input CNF formula
    graph: &'b mut Graph,
    /// State manager that allows to retrieve previous values when backtracking in the search tree
    state: &'b mut StateManager,
    /// The propagator
    propagator: &'b mut Propagator,
    /// component extractor
    component_extractor: &'b mut ComponentExtractor,
}

impl<'b> Preprocessor<'b>
where
{

    pub fn new(graph: &'b mut Graph, state: &'b mut StateManager, propagator: &'b mut Propagator, component_extractor: &'b mut ComponentExtractor) -> Self {
        Self {
            graph,
            state,
            propagator,
            component_extractor,
        }
    }
    
    pub fn preprocess(&mut self) -> Option<Float> {
        let mut p = f128!(1.0);

        for variable in self.graph.variables_iter() {
            if self.graph[variable].is_probabilitic() && self.graph[variable].weight().unwrap() == 1.0 {
                self.propagator.add_to_propagation_stack(variable, true, 0, None);
            }
        }
        
        // Find unit clauses
        for clause in self.graph.clauses_iter() {
            if self.graph[clause].is_unit(self.state) {
                let l = self.graph[clause].get_unit_assigment(self.state);
                self.propagator.add_to_propagation_stack(l.to_variable(), l.is_positive(), 0, None);
            }
        }
        match self.propagator.propagate(self.graph, self.state, ComponentIndex(0), self.component_extractor, 0, true) {
            Err(_) => return None,
            Ok(_) => {
                p *= self.propagator.get_propagation_prob();
            }
        };
        Some(p)
    }
}
