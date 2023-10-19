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
use super::graph::ClauseIndex;
use super::sparse_set::SparseSet;
use super::watched_vector::WatchedVector;
use super::literal::Literal;

use super::graph::{DistributionIndex, VariableIndex, Graph};


#[derive(Debug)]
pub struct Clause {
    id: usize,
    /// If the clause is not learned, this is the literal which is the head of the clause. Otherwise, None
    head: Option<Literal>,
    /// The literals of the clause. Implemented using a vector with watched literals
    literals: WatchedVector,
    /// Vector that stores the children of the clause in the implication graph
    pub children: SparseSet<ClauseIndex>,
    /// Vector that stores the parents of the clause in the implication graph
    pub parents: SparseSet<ClauseIndex>,
    /// Random bitstring used for hash computation
    hash: u64,
    is_learned: bool,
}

impl Clause {

    pub fn new(id: usize, literals: WatchedVector, head: Option<Literal>, is_learned: bool, state: &mut StateManager) -> Self {
        Self {
            id,
            literals,
            head,
            children: SparseSet::new(state),
            parents: SparseSet::new(state),
            hash: rand::random(),
            is_learned
        }
    }
    
    pub fn add_child(&mut self, child: ClauseIndex, state: &mut StateManager) {
        self.children.add(child, state);
    }
    
    pub fn add_parent(&mut self, parent: ClauseIndex, state: &mut StateManager) {
        self.parents.add(parent, state);
    }
    
    pub fn remove_child(&mut self, child: ClauseIndex, state: &mut StateManager) {
        self.children.remove(child, state);
    }
    
    pub fn remove_parent(&mut self, parent: ClauseIndex, state: &mut StateManager) {
        self.parents.remove(parent, state);
    }
    
    pub fn get_bounds_watcher(&self, state: &StateManager) -> (usize, usize) {
        self.literals.get_bounds(state)
    }
    
    pub fn has_probabilistic(&self, state: &StateManager) -> bool {
        self.literals.get_alive_end_watcher(state).is_some()
    }
    
    pub fn set_unconstrained(&self, state: &mut StateManager) {
        self.literals.set_unconstrained(state);
    }
    
    pub fn is_constrained(&self, state: &StateManager) -> bool {
        self.literals.is_constrained(state)
    }
    
    pub fn hash(&self) -> u64 {
        self.hash
    }
    
    pub fn get_constrained_distribution(&self, state: &StateManager, g: &Graph) -> Option<DistributionIndex> {
        match self.literals.get_alive_end_watcher(state) {
            None => None,
            Some(l) => g[l.to_variable()].distribution(),
        }
    }
    
    pub fn number_parents(&self) -> usize {
        self.parents.capacity()
    }
    
    pub fn number_children(&self) -> usize {
        self.children.capacity()
    }
    
    pub fn number_constrained_parents(&self, state: &StateManager) -> usize {
        self.parents.len(state)
    }
    
    pub fn number_constrained_children(&self, state: &StateManager) -> usize {
        self.children.len(state)
    }
    
    pub fn head(&self) -> Option<Literal> {
        self.head
    }

    pub fn is_head(&self, variable: VariableIndex) -> bool {
        match self.head {
            None => false,
            Some(h) => h.to_variable() == variable,
        }
    }
    
    pub fn notify_variable_value(&mut self, variable: VariableIndex, value: bool, probabilistic: bool, state: &mut StateManager) -> VariableIndex {
        if !probabilistic {
            self.literals.update_watcher_start(variable, value, state)
        } else {
            self.literals.update_watcher_end(variable, value, state)
        }
    }

    pub fn is_unit(&self, state: &StateManager) -> bool {
        if !self.is_constrained(state) {
            return false;
        }
        let bounds = self.literals.get_bounds(state);
        bounds.0 + bounds.1 == 1
    }
    
    pub fn get_unit_assigment(&self, state: &StateManager) -> Literal {
        debug_assert!(self.is_unit(state));
        let bounds = self.literals.get_bounds(state);
        if bounds.0 == 0 {
            self.literals[self.literals.limit()]
        } else {
            self.literals[0]
        }
    }
    
    pub fn has_probabilistic_in_body(&self, state: &StateManager) -> bool {
        let bound_probabilistic = self.literals.get_bounds(state).1;
        for i in 0..bound_probabilistic {
            if !self.literals[self.literals.limit() + i].is_positive() {
                return true;
            }
        }
        return false;
    }
    
    pub fn has_deterministic_in_body(&self, state: &StateManager) -> bool {
        let bound_deterministic = self.literals.get_bounds(state).0;
        for i in 0..bound_deterministic {
            if !self.literals[i].is_positive() {
                return true;
            }
        }
        return false;
    }
    
    pub fn is_learned(&self) -> bool {
        self.is_learned
    }
    
    // --- ITERATORRS --- //

    pub fn iter_parents(&self, state: &StateManager) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.parents.iter(state)
    }
    
    pub fn iter_children(&self, state: &StateManager) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.children.iter(state)
    }
    
    pub fn iter(&self) -> impl Iterator<Item = Literal> + '_ {
        self.literals.iter()
    }
    
    pub fn iter_variables(&self) -> impl Iterator<Item = VariableIndex> + '_ {
        self.literals.iter().map(|l| l.to_variable())
    }
    
    pub fn iter_probabilistic_variables(&self) -> impl Iterator<Item = VariableIndex> + '_ {
        self.literals.iter_end().map(|l| l.to_variable())
    }
    
}

impl std::fmt::Display for Clause {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "C{}: {}", self.id + 1, self.literals.iter().map(|l| format!("{}", l)).collect::<Vec<String>>().join(" "))
    }
}