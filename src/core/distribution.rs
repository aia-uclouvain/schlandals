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

//! This module provide the implementation of a distribution in Schlandals.
//! A distribution is a set of variable that respects the following constraints:
//!     1. Every variable must have a weight
//!     2. The sum of the variables' weight must sum to 1
//!     3. In each model of the input formula, exactly one of the variables is set to true

use super::graph::VariableIndex;
use search_trail::{StateManager, ReversibleUsize, UsizeManager, ReversibleBool, BoolManager, ReversibleF64, F64Manager};

/// A distribution of the input problem
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Distribution {
    /// Id of the distribution in the problem
    id: usize,
    /// First variable in the distribution
    pub first: VariableIndex,
    /// Number of variable in the distribution
    pub size: usize,
    /// Number of constrained clauses in which the distribution appears
    pub number_clause_unconstrained: ReversibleUsize,
    /// Number of clauses in which the distribution appears
    number_clause: usize,
    /// Number of variables assigned to F in the distribution
    pub number_false: ReversibleUsize,
    /// 
    constrained: ReversibleBool,
    remaining: ReversibleF64,
    branching_candidate: bool,
}

impl Distribution {
    
    pub fn new(id: usize, first: VariableIndex, size: usize, state: &mut StateManager) -> Self {
        Self {
            id,
            first,
            size,
            number_clause_unconstrained: state.manage_usize(0),
            number_clause: 0,
            number_false: state.manage_usize(0),
            constrained: state.manage_bool(true),
            remaining: state.manage_f64(1.0),
            branching_candidate: true,
        }
    }
    
    /// Increments the number of clauses in which the distribution appears
    pub fn increment_clause(&mut self) {
        self.number_clause += 1;
    }
    
    /// Decrements the number of constrained clause in which the distribution appears. This
    /// operation is reversed when the trail restore its state.
    /// Returns the remaining number of constrained clauses in which it appears.
    pub fn decrement_constrained(&self, state: &mut StateManager) -> usize {
        self.number_clause - state.increment_usize(self.number_clause_unconstrained)
    }
    
    /// Icrements the number of variable assigned to false in the distribution. This operation
    /// is reversed when the trail restore its state.
    pub fn increment_number_false(&self, state: &mut StateManager) -> usize {
        state.increment_usize(self.number_false)
    }
    
    /// Returns the number of unfixed variables in the distribution. This assume that the distribution
    /// has no variable set to true (otherwise there is no need to consider it).
    pub fn number_unfixed(&self, state: &StateManager) -> usize {
        self.size - state.get_usize(self.number_false)
    }
    
    /// Returns the number of variable set to false in the distribution.
    pub fn number_false(&self, state: &StateManager) -> usize {
        state.get_usize(self.number_false)
    }
    
    /// Returns the start of the distribution in the vector of variables in the graph.
    pub fn start(&self) -> VariableIndex {
        self.first
    }
    
    pub fn is_constrained(&self, state: &StateManager) -> bool {
        state.get_bool(self.constrained)
    }

    pub fn set_unconstrained(&self, state: &mut StateManager) {
        state.set_bool(self.constrained, false);
    }

    pub fn remaining(&self, state: &StateManager) -> f64 {
        state.get_f64(self.remaining)
    }

    pub fn remove_probability_mass(&self, removed: f64, state: &mut StateManager) {
        let new_value = state.get_f64(self.remaining) - removed;
        state.set_f64(self.remaining, new_value);
    }

    pub fn set_branching_candidate(&mut self, value: bool) {
        self.branching_candidate = value;
    }

    pub fn is_branching_candidate(&self) -> bool {
        self.branching_candidate
    }

    // --- ITERATOR --- //

    /// Returns an iterator on the variables of the distribution
    pub fn iter_variables(&self) -> impl Iterator<Item = VariableIndex> {
        (self.first.0..(self.first.0 + self.size)).map(VariableIndex)
    }
}

impl std::fmt::Display for Distribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "D{}", self.id + 1)
    }
}
