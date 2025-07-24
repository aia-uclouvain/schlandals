//! This module provide the implementation of a distribution in Schlandals.
//! A distribution is a set of variable that respects the following constraints:
//!     1. Every variable must have a weight
//!     2. The sum of the variables' weight must sum to 1
//!     3. In each model of the input formula, exactly one of the variables is set to true

use super::problem::VariableIndex;
use search_trail::{ReversibleUsize, StateManager, UsizeManager};

/// A distribution of the input problem
#[derive(Debug)]
pub struct Distribution {
    /// Id of the distribution in the problem
    id: usize,
    /// First variable in the distribution
    first: VariableIndex,
    domain_size: usize,
    /// Number of variable in the distribution
    size: ReversibleUsize,
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
            old_first: first,
        }
    }

    pub fn is_constrained(&self, state: &StateManager) -> bool {
        state.get_usize(self.size) > 1
    }

    pub fn set_unconstrained(&self, state: &mut StateManager) {
        state.set_usize(self.size, 0);
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

    pub fn size(&self, state: &StateManager) -> usize {
        state.get_usize(self.size)
    }

    pub fn domain_size(&self) -> usize {
        self.domain_size
    }

    pub fn is_partial_domain(&self, state: &StateManager) -> bool {
        self.domain_size != self.size(state)
    }

    pub fn set_domain_size(&mut self, domain_size: usize) {
        self.domain_size = domain_size;
    }

    pub fn decrement_size(&self, state: &mut StateManager) {
        state.decrement_usize(self.size);
    }

    // --- ITERATOR --- //

    /// Returns an iterator on the variables of the distribution
    pub fn iter_variables(&self) -> impl Iterator<Item = VariableIndex> + use<> {
        (self.first.0..(self.first.0 + self.domain_size)).map(VariableIndex)
    }
}

impl std::fmt::Display for Distribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "D{}", self.id + 1)
    }
}
