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

//! Implementation of a two-literal based watch vector. This vector differ from classical watched vector
//! in the following way.
//! Traditionnal two-watch literal implementaiton imposes that each clause is watched by two of its literals.
//! However, in order to do its additional propagation, Schlandals needs to know when a clause has deterministic
//! variable in its body.
//! Hence, the variable in the vector are sorted by type. First, the deterministic variable and after the probabilistic
//! variables. They are stored in the same vector, but are considered as two separated watched vector.
//! This means that the vector is watched by four literals.
//! The operations are done in a manner such that the first two literals are always wathing the vector

use search_trail::{StateManager, BoolManager, ReversibleBool};
use super::literal::Literal;
use super::problem::VariableIndex;
use rustc_hash::FxHashMap;

#[derive(Debug)]
pub struct WatchedVector {
    /// The literals in the clause that owns the vector
    literals: Vec<Literal>,
    /// The index the the first probabilistic literal in the clause
    limit: usize,
    /// Is the clause constrained
    constrained: ReversibleBool,
}

impl WatchedVector {

    pub fn new(literals: Vec<Literal>, limit: usize, state: &mut StateManager) -> Self {
         Self {
            literals,
            limit,
            constrained: state.manage_bool(true),
        }
    }

    /// Returns the number of literals in the vector
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Returns the number of unfixed deterministic/probabilistic watchers for this vector
    pub fn get_bounds(&self, state: &StateManager) -> (usize, usize) {
        let bound_deterministic = (0..self.limit.min(2)).filter(|i| !self.literals[*i].is_variable_fixed(state)).count();
        let bound_probabilistic = (self.limit..(self.limit + 2).min(self.literals.len())).filter(|i| !self.literals[*i].is_variable_fixed(state)).count();
        (bound_deterministic, bound_probabilistic)
    }
    
    /// Returns true iff the clause that owns the vector is constrained
    pub fn is_constrained(&self, state: &StateManager) -> bool {
        state.get_bool(self.constrained)
    }
    
    /// Set the vector (and the clause that owns it) as unconstrained
    pub fn set_unconstrained(&self, state: &mut StateManager) {
        state.set_bool(self.constrained, false);
    }
    
    /// Update, if possible, the watcher that is not alive anymore and returns the variable that is the new watcher.
    /// If the watcher was not updated, return watcher.
    fn update_watcher(&mut self, watcher: VariableIndex, value: bool, start: usize, end: usize, bound: usize, state: &mut StateManager) -> VariableIndex {
        let id = if self.literals[start].to_variable() == watcher { start } else { (start + 1).min(end - 1)};
        if (self.literals[id].is_positive() && value) || (!self.literals[id].is_positive() && !value) {
            self.set_unconstrained(state);
            return watcher;
        }
        if bound >= 1 {
            if watcher == self.literals[start].to_variable() {
                self.literals.swap(start, start + 1);
            }
            debug_assert!(!self.literals[start].is_variable_fixed(state));
            for i in (start + 2)..end {
                if !self.literals[i].is_variable_fixed(state) {
                    self.literals.swap(start + 1, i);
                    break;
                }
            }
            self.literals[start + 1].to_variable()
        } else {
            debug_assert!(bound == 0);
            self.literals[start].to_variable()
        }
    }
    
    /// Updates the deterministic watchers
    pub fn update_watcher_start(&mut self, watcher: VariableIndex, value: bool, state: &mut StateManager) -> VariableIndex {
        let (bound, _) = self.get_bounds(state);
        self.update_watcher(watcher, value, 0, self.limit, bound, state)
    }

    /// Updates the probabilistic watchers
    pub fn update_watcher_end(&mut self, watcher: VariableIndex, value: bool, state: &mut StateManager) -> VariableIndex {
        let (_, bound) = self.get_bounds(state);
        self.update_watcher(watcher, value, self.limit, self.literals.len(), bound, state)
    }
    
    /// If there exists an unassigned probabilistic variable in the vector, returns it (This is equivalent to looking
    /// at the watchers).
    pub fn get_alive_end_watcher(&self, state: &StateManager) -> Option<Literal> {
        if self.limit >= self.literals.len() {
            return None;
        }
        if self.literals[self.limit].is_variable_fixed(state) {
            None
        } else {
            Some(self.literals[self.limit])
        }
    }
    
    /// Returns the index of the first probabilistic variable in the clause.
    pub fn limit(&self) -> usize {
        self.limit
    }

    pub fn reduce(&mut self, map: &FxHashMap<VariableIndex, VariableIndex>) {
        for i in (0..self.literals.len()).rev() {
            let v = self.literals[i].to_variable();
            match  map.get(&v).copied() {
                Some(new_v) => {
                    let is_pos = self.literals[i].is_positive();
                    let idx = self.literals[i].trail_index();
                    self.literals[i] = Literal::from_variable(new_v, is_pos, idx);
                }
                None => {
                    if i < self.limit {
                        self.literals.remove(i);
                        self.limit -= 1;
                    } else {
                        self.literals.swap_remove(i);
                    }
                },
            };
        }
    }

    pub fn remove(&mut self, variable: VariableIndex) {
        for i in 0..self.literals.len() {
            if variable == self.literals[i].to_variable() {
                if i < self.limit {
                    self.literals.remove(i);
                    self.limit -= 1;
                } else {
                    self.literals.swap_remove(i);
                }
                break;
            }
        }
    }

    pub fn get_watchers(&self) -> Vec<Option<VariableIndex>> {
        let mut watchers: Vec<Option<VariableIndex>> = vec![];
        for i in 0..self.limit.min(2) {
            watchers.push(Some(self.literals[i].to_variable()));
        }
        for _ in watchers.len()..2 {
            watchers.push(None);
        }
        for i in self.limit..(self.limit + 2).min(self.literals.len()) {
            watchers.push(Some(self.literals[i].to_variable()));
        }
        for _ in watchers.len()..4 {
            watchers.push(None);
        }
        watchers
    }

    // --- ITERATOR --- //

    /// Returns an interator on the variable in the vector
    pub fn iter(&self) -> impl Iterator<Item = Literal> + '_ {
        self.literals.iter().copied()
    }
    
    /// Returns an iterator on the probabilistic variables in the vector
    pub fn iter_end(&self) -> impl Iterator<Item = Literal> + '_ {
        self.literals.iter().skip(self.limit).copied()
    }
    
}

impl std::ops::Index<usize> for WatchedVector {
    
    type Output = Literal;

    fn index(&self, index: usize) -> &Self::Output {
        &self.literals[index]
    }
}
