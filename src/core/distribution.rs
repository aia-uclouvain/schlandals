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

use super::{problem::{ClauseIndex, DistributionIndex, VariableIndex}, sparse_set::SparseSet};
use search_trail::{StateManager, ReversibleUsize, UsizeManager, ReversibleF64, F64Manager};
use rustc_hash::FxHashMap;

/// A distribution of the input problem
#[derive(Debug)]
pub struct Distribution {
    /// Id of the distribution in the problem
    id: usize,
    /// First variable in the distribution
    first: VariableIndex,
    /// Number of variable in the distribution
    size: usize,
    /// Reversible sparse set containing the distribution constraining the distribution. We assume
    /// that the distribution do not appear twice in the same clause. At the time of the writing of
    /// these line this is a reasonnable constraint, but time will tell if I must update this code.
    clauses: SparseSet<ClauseIndex>,
    /// Number of variables assigned to F in the distribution
    number_false: ReversibleUsize,
    /// Sum of the weight of the unfixed variables in the distribution
    remaining: ReversibleF64,
    /// Is the distribution a candidate for branching ?
    branching_candidate: bool,
    /// Initial index of the distribution in the problem
    old_index: usize,
    /// Initial first variable of the distribution in the problem
    old_first: VariableIndex,
}

impl Distribution {
    
    pub fn new(id: usize, first: VariableIndex, size: usize, state: &mut StateManager) -> Self {
        Self {
            id,
            first,
            size,
            clauses: SparseSet::new(state),
            number_false: state.manage_usize(0),
            remaining: state.manage_f64(1.0),
            branching_candidate: true,
            old_index: id,
            old_first: first,
        }
    }

    pub fn add_clause(&mut self, clause: ClauseIndex, state: &mut StateManager) {
        self.clauses.add(clause, state);
    }

    pub fn remove_clause(&mut self, clause: ClauseIndex, state: &mut StateManager) {
        self.clauses.remove(clause, state);
    }

    pub fn is_constrained(&self, state: &StateManager) -> bool {
        self.clauses.len(state) != 0
    }

    pub fn set_unconstrained(&self, state: &mut StateManager) {
        self.clauses.remove_all(state);
    }

    /// Icrements the number of variable assigned to false in the distribution. This operation
    /// is reversed when the trail restore its state.
    pub fn increment_number_false(&self, state: &mut StateManager) -> usize {
        state.increment_usize(self.number_false)
    }
    
    /// Returns the number of unfixed variables in the distribution. This assume that the distribution
    /// has no variable set to true (otherwise there is no need to consider it).
    pub fn number_unfixed(&self, state: &StateManager) -> usize {
        let n = state.get_usize(self.number_false);
        if  n > self.size {
            panic!("Size is {} but number false is {}", self.size, n);
        }
        self.size - state.get_usize(self.number_false)
    }
    
    /// Returns the number of variable set to false in the distribution.
    pub fn number_false(&self, state: &StateManager) -> usize {
        state.get_usize(self.number_false)
    }

    /// Returns the initial index of the distribution in the problem
    pub fn old_index(&self) -> DistributionIndex {
        DistributionIndex(self.old_index)
    }

    /// Returns the initial first variable of the distribution in the problem
    pub fn old_first(&self) -> VariableIndex {
        self.old_first
    }
    
    /// Returns the start of the distribution in the vector of variables in the problem.
    pub fn start(&self) -> VariableIndex {
        self.first
    }

    pub fn set_start(&mut self, start: VariableIndex) {
        self.first = start;
    }

    pub fn set_size(&mut self, size: usize) {
        self.size = size;
    }
    
    pub fn remaining(&self, state: &StateManager) -> f64 {
        state.get_f64(self.remaining)
    }

    pub fn remove_probability_mass(&self, removed: f64, state: &mut StateManager) {
        let old_value = state.get_f64(self.remaining);
        let new_value = old_value- removed;
        state.set_f64(self.remaining, new_value);
    }

    pub fn set_branching_candidate(&mut self, value: bool) {
        self.branching_candidate = value;
    }

    pub fn is_branching_candidate(&self) -> bool {
        self.branching_candidate
    }

    pub fn size(&self) -> usize { self.size }

    pub fn update_clauses(&mut self, map: &FxHashMap<ClauseIndex, ClauseIndex>, state: &mut StateManager) {
        self.clauses.clear(map, state);
    }

    // --- ITERATOR --- //

    /// Returns an iterator on the variables of the distribution
    pub fn iter_variables(&self) -> impl Iterator<Item = VariableIndex> {
        (self.first.0..(self.first.0 + self.size)).map(VariableIndex)
    }

    pub fn iter_clauses(&self, state: &StateManager) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.clauses.iter(state)
    }
}

impl std::fmt::Display for Distribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "D{}", self.id + 1)
    }
}
