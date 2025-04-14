//! Representation of a clause in Schlandals. All clauses used in Schlandals are Horn clause, which
//! means that they have at most one positive literal, the head of the clause.
//! The literals of the clause (head included) are stored in a vector that implements the 2-watch literals
//! method.
//! However, the specific needs of Schlandals for the propagation impose that each clause is watched by two pairs
//! of watched literals.
//! One pair is composed of deterministic literals, and the other of probabilistic ones.
//! In this way the propagator can, at any time, query a boud on the number of unfixed deterministic/probabilistic
//! variables in the clause.

use search_trail::{BoolManager, ReversibleBool, UsizeManager, ReversibleUsize, StateManager};
use super::problem::ClauseIndex;
use super::sparse_set::SparseSet;
use super::literal::Literal;
use rustc_hash::FxHashMap;

use super::problem::{DistributionIndex, VariableIndex, Problem};

#[derive(Debug)]
pub struct Clause {
    /// id of the clause in the input problem
    id: usize,
    /// The literals of the clause. Implemented using a vector with watched literals
    literals: Vec<Literal>,
    /// Vector that stores the children of the clause in the implication problem
    pub children: SparseSet<ClauseIndex>,
    /// Vector that stores the parents of the clause in the implication problem
    pub parents: SparseSet<ClauseIndex>,
    /// Random bitstring used for hash computation
    hash: u64,
    /// Has the clause been learned during the search
    is_learned: bool,
    /// Is the clause active (i.e., not yet satisfied)
    active: ReversibleBool,
    /// Number of deterministic variables in the body of the clause
    number_deterministic_in_body: ReversibleUsize,
    is_head_f_reachable: ReversibleBool,
}

impl Clause {

    pub fn new(id: usize, literals: Vec<Literal>, number_deterministic_in_body: usize, is_learned: bool, state: &mut StateManager) -> Self {
        Self {
            id,
            literals,
            children: SparseSet::new(state),
            parents: SparseSet::new(state),
            hash: rand::random(),
            is_learned,
            active: state.manage_bool(true),
            number_deterministic_in_body: state.manage_usize(number_deterministic_in_body),
            is_head_f_reachable: state.manage_bool(false),
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
    
    /// Set the clause as unconstrained. This operation is reverted when the state manager restore its state.
    pub fn deactivate(&self, state: &mut StateManager) {
        state.set_bool(self.active, false);
    }
    
    /// Returns true iff the clause is constrained
    pub fn is_active(&self, state: &StateManager) -> bool {
        state.get_bool(self.active)
    }
    
    /// Returns the hash of the clause
    pub fn hash(&self) -> u64 {
        self.hash
    }
    
    /// If the clause still has unfixed probabilisitc variables, return the distribution of the first watcher.
    /// Else, return None.
    pub fn get_constrained_distribution(&self, state: &StateManager, p: &Problem) -> Option<DistributionIndex> {
        self.literals.iter().filter(|l| {
            let v = l.to_variable();
            !p[v].is_fixed(state) && p[v].is_probabilitic()
        }).map(|l| p[l.to_variable()].distribution().unwrap()).next()
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
    
    /// Notify the clause that the given variable has taken the given value. Updates the watchers accordingly.
    pub fn notify_variable_value(&mut self, variable: VariableIndex, state: &mut StateManager) -> VariableIndex {
        if self.literals[0].to_variable() == variable {
            self.literals.swap(0, 1);
        }
        for i in 2..self.literals.len() {
            if !self.literals[i].is_variable_fixed(state) {
                self.literals.swap(1, i);
                break;
            }
        }
        self.literals[1].to_variable()
    }

    /// Returns true iff the clause is unit
    pub fn is_unit(&self, state: &StateManager) -> bool {
        if !self.is_active(state) {
            return false;
        }
        if self.literals.len() == 1 {
            return true;
        }
        !self.literals[0].is_variable_fixed(state) && self.literals[1].is_variable_fixed(state)
    }

    /// Returns the last unfixed literal in the unit clause
    pub fn get_unit_assigment(&self, state: &StateManager) -> Literal {
        debug_assert!(self.is_unit(state));
        self.literals[0]
    }

    pub fn refresh_number_deterministic_in_body(&self, number: usize,  state: &mut StateManager) {
        state.set_usize(self.number_deterministic_in_body, number);
    }

    pub fn decrement_deterministic_in_body(&self, state: &mut StateManager) {
        state.decrement_usize(self.number_deterministic_in_body);
    }

    /// Returns true iff the clause stil has unfixed deterministic variables in its body
    pub fn has_deterministic_in_body(&self, state: &StateManager) -> bool {
        state.get_usize(self.number_deterministic_in_body) > 0
    }

    pub fn set_head_f_reachable(&self, state: &mut StateManager) {
        state.set_bool(self.is_head_f_reachable, true);
    }

    pub fn is_head_f_reachable(&self, state: &StateManager) -> bool {
        state.get_bool(self.is_head_f_reachable)
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
    pub fn iter(&self) -> impl Iterator<Item = Literal> + use<'_> {
        self.literals.iter().copied()
    }
    
    /// Returns an iterator on the variables represented by the literals of the clause
    pub fn iter_variables(&self) -> impl Iterator<Item = VariableIndex> + '_ {
        self.literals.iter().map(|l| l.to_variable())
    }
    
    pub fn clear(&mut self, map: &FxHashMap<ClauseIndex, ClauseIndex>, state: &mut StateManager) {
        self.children.clear(map, state);
        self.parents.clear(map, state);
    }

    pub fn clear_literals(&mut self, map: &FxHashMap<VariableIndex, VariableIndex>) {
        for i in (0..self.literals.len()).rev() {
            let v = self.literals[i].to_variable();
            match map.get(&v).copied() {
                Some(new_v) => {
                    self.literals[i].update_variable(new_v);
                },
                None => {
                    self.literals.swap_remove(i);
                }
            }
        }
    }

    pub fn get_watchers(&self) -> Vec<VariableIndex> {
        self.literals.iter().take(2).map(|l| l.to_variable()).collect()
    }
}

// Writes a clause as C{id}: l1 l2 ... ln
impl std::fmt::Display for Clause {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "C{}: {}", self.id + 1, self.literals.iter().map(|l| format!("{}", l)).collect::<Vec<String>>().join(" "))
    }
}
