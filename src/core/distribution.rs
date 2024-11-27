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

use super::{problem::{ClauseIndex, VariableIndex}, sparse_set::SparseSet};
use search_trail::{F64Manager, ReversibleF64, ReversibleUsize, StateManager, UsizeManager};
use rustc_hash::FxHashMap;
use malachite::Rational;
use malachite::num::conversion::traits::RoundingFrom;
use malachite::rounding_modes::RoundingMode::Nearest;

/// A distribution of the input problem
#[derive(Debug)]
pub struct Distribution {
    /// Id of the distribution in the problem
    id: usize,
    /// First variable in the distribution
    first: VariableIndex,
    pub domain_size: usize,
    /// Number of variable in the distribution
    size: ReversibleUsize,
    /// Reversible sparse set containing the distribution constraining the distribution. We assume
    /// that the distribution do not appear twice in the same clause. At the time of the writing of
    /// these line this is a reasonnable constraint, but time will tell if I must update this code.
    clauses: SparseSet<ClauseIndex>,
    /// Sum of the weight of the unfixed variables in the distribution
    remaining: ReversibleF64,
    /// Initial first variable of the distribution in the problem
    old_first: VariableIndex,
}

impl Distribution {
    
    pub fn new(id: usize, first: VariableIndex, size: usize, state: &mut StateManager) -> Self {
        Self {
            id,
            first,
            domain_size: size,
            size: state.manage_usize(size),
            clauses: SparseSet::new(state),
            remaining: state.manage_f64(1.0),
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
    
    /// Returns the initial index of the distribution in the problem
    pub fn old_index(&self) -> usize {
        self.id
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

    pub fn remaining(&self, state: &StateManager) -> f64 {
        state.get_f64(self.remaining)
    }

    pub fn set_remaining(&self, value: Rational, state: &mut StateManager) {
        let remaining = f64::rounding_from(value, Nearest).0;
        state.set_f64(self.remaining, remaining);
    }

    pub fn remove_probability_mass(&self, removed: Rational, state: &mut StateManager) {
        let old_value = state.get_f64(self.remaining);
        let new_value = old_value - f64::rounding_from(removed, Nearest).0;
        state.set_f64(self.remaining, new_value);
        let old_size = state.get_usize(self.size);
        state.set_usize(self.size, old_size - 1);
    }

    pub fn size(&self, state: &StateManager) -> usize {
        state.get_usize(self.size)
    }

    pub fn update_clauses(&mut self, map: &FxHashMap<ClauseIndex, ClauseIndex>, state: &mut StateManager) {
        self.clauses.clear(map, state);
    }

    pub fn set_domain_size(&mut self, domain_size: usize) {
        self.domain_size = domain_size;
    }

    // --- ITERATOR --- //

    /// Returns an iterator on the variables of the distribution
    pub fn iter_variables(&self) -> impl Iterator<Item = VariableIndex> {
        (self.first.0..(self.first.0 + self.domain_size)).map(VariableIndex)
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
