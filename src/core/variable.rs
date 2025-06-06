//! An implementation of a variable in Schlandals. A variable is the core unit
//! for reasoning in schlandals as they define the distributions.

use search_trail::*;
use crate::core::problem::{ClauseIndex, DistributionIndex};
use rustc_hash::FxHashMap;
use malachite::rational::Rational;
use super::sparse_set::SparseSet;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum Reason {
    Clause(ClauseIndex),
    Distribution(DistributionIndex),
}

/// Data structure that actually holds the data of a  variable of the input problem
#[derive(Debug)]
pub struct Variable {
    /// The id of the variable in the input problem
    id: usize,
    /// The weight of the variable. None if the variable is deterministic
    weight: Option<Rational>,
    /// Index of the variable in the distribution (from the initial problem, after sorting them by
    /// weight)
    index_in_distribution: Option<usize>,
    /// The distribution in which the variable is.  None if the variable is deterministic
    distribution: Option<DistributionIndex>,
    /// The clauses in which the variable appears with positive polarity
    clauses_positive: SparseSet<ClauseIndex>,
    /// The clauses in which the variable appears with negative polarity
    clauses_negative: SparseSet<ClauseIndex>,
    /// The value assigned to the variable
    value: ReversibleOptionBool,
    /// Level at which the decision was made for this variable
    decision: isize,
    /// Index in the assignment stack at which the decision has been made for the variable
    assignment_position: ReversibleUsize,
    /// The clause that set the variable, if any
    reason: Option<Reason>,
    /// True if the variable has been implied during BUP
    is_implied: ReversibleBool,
    /// Random u64 associated to the variable, used for hash computation
    hash: u64,
}

impl Variable {
    
    pub fn new(id: usize, weight: Option<Rational>, index_in_distribution: Option<usize>, distribution: Option<DistributionIndex>, state: &mut StateManager) -> Self {
        Self {
            id,
            weight,
            index_in_distribution,
            distribution,
            clauses_positive: SparseSet::new(state),
            clauses_negative: SparseSet::new(state),
            value: state.manage_option_bool(None),
            decision: -1,
            assignment_position: state.manage_usize(0),
            reason: None,
            is_implied: state.manage_bool(false),
            hash: rand::random(),
        }
    }
    
    /// Sets the weight of the variable to the given value
    pub fn set_weight(&mut self, weight: Rational) {
        self.weight = Some(weight);
    }

    /// Sets the index of the variable in the distribution
    pub fn set_distribution_index(&mut self, index: usize) {
        self.index_in_distribution = Some(index);
    }
    
    /// Set the distribution of the variable to the given distribution
    pub fn set_distribution(&mut self, distribution: DistributionIndex) {
        self.distribution = Some(distribution);
    }
    
    /// Returns the distribution of the variable
    pub fn distribution(&self) -> Option<DistributionIndex> {
        self.distribution
    }
    
    /// Returns true iff the variable is probabilistic
    pub fn is_probabilitic(&self) -> bool {
        self.weight.is_some()
    }
    
    /// Returns the weight of the variable
    pub fn weight(&self) -> Option<Rational> {
        self.weight.clone()
    }

    pub fn index_in_distribution(&self) -> Option<usize> {
        self.index_in_distribution
    }

    /// Returns the initial index of the variable in the problem
    pub fn old_index(&self) -> usize {
        self.id
    }
    
    /// Sets the variable to the given value. This operation is reverted when
    /// the trail is restored
    pub fn set_value(&self, value: bool, state: &mut StateManager) {
        state.set_option_bool(self.value, value);
    }
    
    /// Returns the value of the variable
    pub fn value(&self, state: &StateManager) -> Option<bool> {
        state.get_option_bool(self.value)
    }
    
    /// Returns the reversible boolean representing the value assignment
    /// of the variable
    pub fn get_value_index(&self) -> ReversibleOptionBool {
        self.value
    }
    
    /// Returns true iff the variable is fixed
    pub fn is_fixed(&self, state: &StateManager) -> bool {
        state.get_option_bool(self.value).is_some()
    }
    
    /// Adds the clause in the positive occurence list
    pub fn add_clause_positive_occurence(&mut self, clause: ClauseIndex, state: &mut StateManager) {
        self.clauses_positive.add(clause, state);
    }
    
    /// Adds the clause in the negative occurence list
    pub fn add_clause_negative_occurence(&mut self, clause: ClauseIndex, state: &mut StateManager) {
        self.clauses_negative.add(clause, state);
    }
    
    /// Sets the decision level for the variable to the given level
    pub fn set_decision_level(&mut self, level: isize) {
        self.decision = level
    }
    
    /// Returns the decision level for the variable. This function assume that the query is done
    /// only on fixed variable since the level is not reversible. Since this function is used in
    /// clause learning, it should always be the case
    pub fn decision_level(&self) -> isize {
        self.decision
    }
    
    /// Sets the reason of the variable. The reason is either a clause or a distribution which forced,
    /// during boolean unit propagation, the variable to take a given value.
    pub fn set_reason(&mut self, reason: Option<Reason>, state: &mut StateManager) {
        if reason.is_some() {
            state.set_bool(self.is_implied, true);
        } else {
            state.set_bool(self.is_implied, false);
        }
        self.reason = reason;
    }
    
    /// Returns the reason, if any, of the variable.
    pub fn reason(&self, state: &StateManager) -> Option<Reason> {
        if !state.get_bool(self.is_implied) {
            None
        } else {
            self.reason
        }
    }
    
    /// Returns the hash of the variable
    pub fn hash(&self) -> u64 {
        self.hash
    }
    
    /// Sets the assignment position (in the assignment stack) of the variable to the given value
    pub fn set_assignment_position(&self, position: usize, state: &mut StateManager) {
        state.set_usize(self.assignment_position, position);
    }
    
    /// Returns the assignment position (in the assignment stack) of the variable
    pub fn get_assignment_position(&self, state: &StateManager) -> usize {
        state.get_usize(self.assignment_position)
    }

    pub fn number_clauses(&self, state: &StateManager) -> usize {
        self.clauses_positive.len(state) + self.clauses_negative.len(state)
    }
    
    // --- ITERATOR --- //

    /// Returns an iterator on the clauses in which the variable appears with a positive polarity
    pub fn iter_clauses_positive_occurence(&self, state: &StateManager) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.clauses_positive.iter(state)
    }
    
    /// Returns an iterator on the clauses in which the variable appears with a negative polarity
    pub fn iter_clauses_negative_occurence(&self, state: &StateManager) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.clauses_negative.iter(state)
    }

    pub fn clear_clauses(&mut self, map: &FxHashMap<ClauseIndex, ClauseIndex>, state: &mut StateManager) -> usize {
        self.clauses_positive.clear(map, state);
        self.clauses_negative.clear(map, state);
        self.clauses_positive.len(state) + self.clauses_negative.len(state)
    }
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "V{}", self.id + 1)
    }
}
