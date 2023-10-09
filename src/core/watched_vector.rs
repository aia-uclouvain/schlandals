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

use search_trail::{StateManager, ReversibleUsize, UsizeManager, BoolManager, ReversibleBool};
use super::literal::Literal;
use super::graph::VariableIndex;

#[derive(Debug)]
pub struct WatchedVector {
    literals: Vec<Literal>,
    number_watchers_start: ReversibleUsize,
    number_watchers_end: ReversibleUsize,
    limit: usize,
    constrained: ReversibleBool,
}

impl WatchedVector {

    pub fn new(literals: Vec<Literal>, limit: usize,  state: &mut StateManager) -> Self {
        let number_watchers_start = limit.min(2);
        let number_watchers_end = (literals.len() - limit).min(2);
        let vector = Self {
            literals,
            number_watchers_start: state.manage_usize(number_watchers_start),
            number_watchers_end: state.manage_usize(number_watchers_end),
            limit,
            constrained: state.manage_bool(true),
        };
        vector
    }

    pub fn len(&self) -> usize {
        self.literals.len()
    }
    
    pub fn get_bounds(&self, state: &StateManager) -> (usize, usize) {
        (state.get_usize(self.number_watchers_start), state.get_usize(self.number_watchers_end))
    }
    
    pub fn is_constrained(&self, state: &StateManager) -> bool {
        state.get_bool(self.constrained)
    }
    
    pub fn set_unconstrained(&self, state: &mut StateManager) {
        state.set_bool(self.constrained, false);
    }
    
    pub fn update_watcher_start(&mut self, watcher: VariableIndex, value: bool, state: &mut StateManager) -> VariableIndex {
        debug_assert!(state.get_usize(self.number_watchers_start) != 0);
        // The vector is watched by the start literals, so there must be at least one alive.
        // If this was the last deterministic variables watching the vector, then we do not need to
        // modify anything as when we will backtrack, we need it to still watch the literal.
        
        // If the clause is respected, then set it as unconstrained
        let id = if self.literals[0].to_variable() == watcher { 0 } else { 1 };
        if (self.literals[id].is_positive() && value) || (!self.literals[id].is_positive() && !value) {
            self.set_unconstrained(state);
            return watcher;
        }
        if state.get_usize(self.number_watchers_start) == 2 {
            // The other watcher is still alive, we put it in first position
            if watcher == self.literals[0].to_variable() {
                self.literals.swap(0, 1);
            }
            // Now we need to find a replacement for the dead watcher
            let mut new_watcher_found = false;
            for i in 2..self.limit {
                // If the variable is not fixed, new variable to watch !
                if !self.literals[i].is_variable_fixed(state) {
                    new_watcher_found = true;
                    self.literals.swap(1, i);
                }
            }
            if !new_watcher_found {
                state.decrement_usize(self.number_watchers_start);
            }
            self.literals[1].to_variable()
        } else {
            state.decrement_usize(self.number_watchers_start);
            self.literals[0].to_variable()
        }
    }

    pub fn update_watcher_end(&mut self, watcher: VariableIndex, value: bool, state: &mut StateManager) -> VariableIndex {
        debug_assert!(state.get_usize(self.number_watchers_end) != 0);
        let end = self.literals.len() - 1;
        let before_end = self.literals.len() - 2;
        // The vector is watched by the start literals, so there must be at least one alive.
        // If this was the last deterministic variables watching the vector, then we do not need to
        // modify anything as when we will backtrack, we need it to still watch the literal.

        // If the clause is respected, then set it as unconstrained
        let id = if self.literals[end].to_variable() == watcher { end } else { before_end };
        if (self.literals[id].is_positive() && value) || (!self.literals[id].is_positive() && !value) {
            self.set_unconstrained(state);
            return watcher;
        }
        if state.get_usize(self.number_watchers_end) == 2 {
            // The other watcher is still alive, we put it in first position
            if watcher == self.literals[end].to_variable() {
                self.literals.swap(before_end, end);
            }
            debug_assert!(self.literals[before_end].to_variable() == watcher);
            let mut new_watcher_found = false;
            // Now we need to find a replacement for the dead watcher
            for i in (self.limit..self.literals.len() - 1).rev() {
                // If the variable is not fixed, new variable to watch !
                if !self.literals[i].is_variable_fixed(state) {
                    new_watcher_found = true;
                    self.literals.swap(before_end, i);
                }
            }
            if !new_watcher_found {
                state.decrement_usize(self.number_watchers_end);
            }
            self.literals[before_end].to_variable()
        } else {
            debug_assert!(self.literals.last().unwrap().to_variable() == watcher);
            state.decrement_usize(self.number_watchers_end);
            self.literals.last().unwrap().to_variable()
        }
    }
    
    pub fn get_alive_end_watcher(&self, state: &StateManager) -> Option<Literal> {
        if state.get_usize(self.number_watchers_end) == 0 {
            None
        } else {
            self.literals.last().copied()
        }
    }

    // --- ITERATOR --- //

    pub fn iter(&self) -> impl Iterator<Item = Literal> + '_ {
        self.literals.iter().copied()
    }
    
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