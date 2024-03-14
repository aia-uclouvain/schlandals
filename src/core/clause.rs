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

//! Representation of a clause in Schlandals. All clauses used in Schlandals are Horn clause, which
//! means that they have at most one positive literal, the head of the clause.
//! The literals of the clause (head included) are stored in a vector that implements the 2-watch literals
//! method.
//! However, the specific needs of Schlandals for the propagation impose that each clause is watched by two pairs
//! of watched literals.
//! One pair is composed of deterministic literals, and the other of probabilistic ones.
//! In this way the propagator can, at any time, query a boud on the number of unfixed deterministic/probabilistic
//! variables in the clause.

use search_trail::StateManager;
use super::graph::ClauseIndex;
use super::sparse_set::SparseSet;
use super::watched_vector::WatchedVector;
use super::literal::Literal;
use rustc_hash::FxHashMap;

use super::graph::{DistributionIndex, VariableIndex, Graph};

#[derive(Debug)]
pub struct Clause {
    /// id of the clause in the input problem
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
    /// Has the clause been learned during the search
    is_learned: bool,
    in_degree: usize,
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
            is_learned,
            in_degree: 0,
        }
    }
    
    /// Adds a child to the current clause. The child clause has the head of the current clause
    /// as a negative literals in it.
    pub fn add_child(&mut self, child: ClauseIndex, state: &mut StateManager) {
        self.children.add(child, state);
    }
    
    /// Adds a parent to the current clause. The parent clause has its head as a negative literal
    /// in the current clause.
    pub fn add_parent(&mut self, parent: ClauseIndex, state: &mut StateManager) {
        self.parents.add(parent, state);
    }
    
    /// Remove a child from the children of this clause. Not that this operation is reverted when
    /// the state manager restore the state
    pub fn remove_child(&mut self, child: ClauseIndex, state: &mut StateManager) {
        self.children.remove(child, state);
    }
    
    /// Remove a parent from the parents of this clause. Not that this operation is reverted when
    /// the state manager restore the state
    pub fn remove_parent(&mut self, parent: ClauseIndex, state: &mut StateManager) {
        self.parents.remove(parent, state);
    }
    
    /// Returns a bound on the number ofdeterministic (first element) and probabilistic (second element) 
    /// unfixed variable in the clause
    pub fn get_bounds_watcher(&self, state: &StateManager) -> (usize, usize) {
        self.literals.get_bounds(state)
    }
    
    /// Returns true if the clause still has unfixed probabilistic variables in it
    pub fn has_probabilistic(&self, state: &StateManager) -> bool {
        self.literals.get_alive_end_watcher(state).is_some()
    }
    
    /// Set the clause as unconstrained. This operation is reverted when the state manager restore its state.
    pub fn set_unconstrained(&self, state: &mut StateManager) {
        self.literals.set_unconstrained(state);
    }
    
    /// Returns true iff the clause is constrained
    pub fn is_constrained(&self, state: &StateManager) -> bool {
        self.literals.is_constrained(state)
    }
    
    /// Returns the hash of the clause
    pub fn hash(&self) -> u64 {
        self.hash
    }
    
    /// If the clause still has unfixed probabilisitc variables, return the distribution of the first watcher.
    /// Else, return None.
    pub fn get_constrained_distribution(&self, state: &StateManager, g: &Graph) -> Option<DistributionIndex> {
        match self.literals.get_alive_end_watcher(state) {
            None => None,
            Some(l) => g[l.to_variable()].distribution(),
        }
    }
    
    /// Returns the number of parents of the clause in the initial problem
    pub fn number_parents(&self) -> usize {
        self.parents.capacity()
    }
    
    /// Returns the number of children of the clause in the initial problem
    pub fn number_children(&self) -> usize {
        self.children.capacity()
    }
    
    /// Returns the number of constrained parent of the clause
    pub fn number_constrained_parents(&self, state: &StateManager) -> usize {
        self.parents.len(state)
    }
    
    /// Returns the number of constrained children of the clause
    pub fn number_constrained_children(&self, state: &StateManager) -> usize {
        self.children.len(state)
    }
    
    /// Return the head of the clause
    pub fn head(&self) -> Option<Literal> {
        self.head
    }

    /// Returns true iff the variable is the head of the clause
    pub fn is_head(&self, variable: VariableIndex) -> bool {
        match self.head {
            None => false,
            Some(h) => h.to_variable() == variable,
        }
    }
    
    /// Notify the clause that the given variable has taken the given value. Updates the watchers accordingly.
    pub fn notify_variable_value(&mut self, variable: VariableIndex, value: bool, probabilistic: bool, state: &mut StateManager) -> VariableIndex {
        if !probabilistic {
            self.literals.update_watcher_start(variable, value, state)
        } else {
            self.literals.update_watcher_end(variable, value, state)
        }
    }

    /// Returns true iff the clause is unit
    pub fn is_unit(&self, state: &StateManager) -> bool {
        if !self.is_constrained(state) {
            return false;
        }
        let bounds = self.literals.get_bounds(state);
        bounds.0 + bounds.1 == 1
    }
    
    /// Returns the last unfixed literal in the unit clause
    pub fn get_unit_assigment(&self, state: &StateManager) -> Literal {
        debug_assert!(self.is_unit(state));
        let bounds = self.literals.get_bounds(state);
        if bounds.0 == 0 {
            self.literals[self.literals.limit()]
        } else {
            self.literals[0]
        }
    }
    
    /// Returns true iff the clause stil has unfixed deterministic variables in its body
    pub fn has_deterministic_in_body(&self, state: &StateManager) -> bool {
        let bound_deterministic = self.literals.get_bounds(state).0;
        for i in 0..bound_deterministic {
            if !self.literals[i].is_positive() {
                return true;
            }
        }
        return false;
    }
    
    /// Returns true iff the clause is learned
    pub fn is_learned(&self) -> bool {
        self.is_learned
    }
    
    // --- ITERATORRS --- //

    /// Returns an iterator on the (constrained) parents of the clause
    pub fn iter_parents(&self, state: &StateManager) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.parents.iter(state)
    }
    
    /// Returns an iterator on the (constrained) children of the clause
    pub fn iter_children(&self, state: &StateManager) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.children.iter(state)
    }
    
    /// Returns an interator on the literals of the clause
    pub fn iter(&self) -> impl Iterator<Item = Literal> + '_ {
        self.literals.iter()
    }
    
    /// Returns an iterator on the variables represented by the literals of the clause
    pub fn iter_variables(&self) -> impl Iterator<Item = VariableIndex> + '_ {
        self.literals.iter().map(|l| l.to_variable())
    }
    
    /// Returns an iterator on the probabilistic varaibles in the clause
    pub fn iter_probabilistic_variables(&self) -> impl Iterator<Item = VariableIndex> + '_ {
        self.literals.iter_end().map(|l| l.to_variable())
    }

    pub fn clear(&mut self, map: &FxHashMap<ClauseIndex, ClauseIndex>, state: &mut StateManager) {
        self.children.clear(map, state);
        self.parents.clear(map, state);
    }

    pub fn clear_literals(&mut self, map: &FxHashMap<VariableIndex, VariableIndex>) {
        if let Some(lit) = self.head() {
            let variable = lit.to_variable();
            if let Some(new_variable) = map.get(&variable).copied() {
                let pos = lit.is_positive();
                let idx = lit.trail_index();
                self.head = Some(Literal::from_variable(new_variable, pos, idx));
            } else {
                self.head = None;
            }
        }
        self.literals.reduce(map);
    }

    pub fn remove_literals(&mut self, variable: VariableIndex) {
        self.literals.remove(variable);
    }

    pub fn get_watchers(&self) -> Vec<Option<VariableIndex>> {
        self.literals.get_watchers()
    }

    pub fn increment_in_degree(&mut self) {
        self.in_degree += 1;
    }

    pub fn in_degree(&self) -> usize {
        self.in_degree
    }
    
}

// Writes a clause as C{id}: l1 l2 ... ln
impl std::fmt::Display for Clause {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "C{}: {}", self.id + 1, self.literals.iter().map(|l| format!("{}", l)).collect::<Vec<String>>().join(" "))
    }
}
