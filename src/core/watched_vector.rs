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

use search_trail::{StateManager, BoolManager, ReversibleBool};
use super::literal::Literal;
use super::graph::VariableIndex;

#[derive(Debug)]
pub struct WatchedVector {
    literals: Vec<Literal>,
    limit: usize,
    constrained: ReversibleBool,
}

impl WatchedVector {

    pub fn new(literals: Vec<Literal>, limit: usize, state: &mut StateManager) -> Self {
        let vector = Self {
            literals,
            limit,
            constrained: state.manage_bool(true),
        };
        vector
    }

    pub fn len(&self) -> usize {
        self.literals.len()
    }
    
    pub fn get_bounds(&self, state: &StateManager) -> (usize, usize) {
        let bound_deterministic = (0..self.limit.min(2)).filter(|i| !self.literals[*i].is_variable_fixed(state)).count();
        let bound_probabilistic = (self.limit..(self.limit + 2).min(self.literals.len())).filter(|i| !self.literals[*i].is_variable_fixed(state)).count();
        (bound_deterministic, bound_probabilistic)
    }
    
    pub fn is_constrained(&self, state: &StateManager) -> bool {
        state.get_bool(self.constrained)
    }
    
    pub fn set_unconstrained(&self, state: &mut StateManager) {
        state.set_bool(self.constrained, false);
    }
    
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
    
    pub fn update_watcher_start(&mut self, watcher: VariableIndex, value: bool, state: &mut StateManager) -> VariableIndex {
        let (bound, _) = self.get_bounds(state);
        self.update_watcher(watcher, value, 0, self.limit, bound, state)
    }

    pub fn update_watcher_end(&mut self, watcher: VariableIndex, value: bool, state: &mut StateManager) -> VariableIndex {
        let (_, bound) = self.get_bounds(state);
        self.update_watcher(watcher, value, self.limit, self.literals.len(), bound, state)
    }
    
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
    
    pub fn limit(&self) -> usize {
        self.limit
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