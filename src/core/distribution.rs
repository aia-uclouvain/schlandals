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

use super::graph::VariableIndex;
use search_trail::{StateManager, ReversibleUsize, UsizeManager};

/// Represents a set of variable in a same distribution. This assume that the variable of a distribution
/// are inserted in the graph one after the other (i.e. that their `VariableIndex` are consecutive).
/// Since no variable should be removed from the graph once constructed, this should not be a problem.
/// Thus a distribution is identified by the first `VariableIndex` and the number of variable in it.
/// 
/// We also store the number of clauses in which the distribution appears. This is usefull to compute
/// the unconstrained probability during the propagation.
/// In the same manner, the number of variable assigned to false in the distribution is kept, allowing us
/// to efficiently detect that only one variable remains in a distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Distribution {
    id: usize,
    /// First variable in the distribution
    pub first: VariableIndex,
    /// Number of variable in the distribution
    pub size: usize,
    /// Number of constrained clauses in which the distribution appears
    pub number_clause_unconstrained: ReversibleUsize,
    number_clause: usize,
    /// Number of variables assigned to F in the distribution
    pub number_false: ReversibleUsize,
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
        }
    }
    
    pub fn increment_clause(&mut self) {
        self.number_clause += 1;
    }
    
    pub fn decrement_constrained(&self, state: &mut StateManager) -> usize {
        self.number_clause - state.increment_usize(self.number_clause_unconstrained)
    }
    
    pub fn increment_number_false(&self, state: &mut StateManager) -> usize {
        state.increment_usize(self.number_false)
    }
    
    pub fn number_unfixed(&self, state: &StateManager) -> usize {
        self.size - state.get_usize(self.number_false)
    }
    
    pub fn number_false(&self, state: &StateManager) -> usize {
        state.get_usize(self.number_false)
    }
    
    pub fn start(&self) -> VariableIndex {
        self.first
    }
    
    // --- ITERATOR --- //

    pub fn iter_variables(&self) -> impl Iterator<Item = VariableIndex> {
        (self.first.0..(self.first.0 + self.size)).map(VariableIndex)
    }
}

impl std::fmt::Display for Distribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "D{}", self.id + 1)
    }
}
