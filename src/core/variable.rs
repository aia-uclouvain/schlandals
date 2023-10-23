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

use search_trail::*;
use crate::core::graph::{ClauseIndex, DistributionIndex};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum Reason {
    Clause(ClauseIndex),
    Distribution(DistributionIndex),
}

/// Data structure that actually holds the data of a  variable of the input problem
#[derive(Debug)]
pub struct Variable {
    id: usize,
    /// If `probabilistic` is `true`, then this is the weight associated to the variable. Otherwise,
    /// this is None.
    weight: Option<f64>,
    /// If `probabilistic` is `true`, this is the index of the distribution containing this node. Otherwise,
    /// this is None.
    distribution: Option<DistributionIndex>,
    /// The clauses in which the variable appears with positive polarity
    clauses_positive: Vec<ClauseIndex>,
    /// The clauses in which the variable appears with negative polarity
    clauses_negative: Vec<ClauseIndex>,
    /// The value assigned to the variable
    value: ReversibleOptionBool,
    /// Level at which the decision was made for this variable
    decision: isize,
    assignment_position: ReversibleUsize,
    /// The clause that set the variable, if any
    reason: Option<Reason>,
    is_implied: ReversibleBool,
    /// Random u64 associated to the variable, used for hash computation
    hash: u64,
    marked: bool,
}

impl Variable {
    
    pub fn new(id: usize, weight: Option<f64>, distribution: Option<DistributionIndex>, state: &mut StateManager) -> Self {
        Self {
            id,
            weight,
            distribution,
            clauses_positive: vec![],
            clauses_negative: vec![],
            value: state.manage_option_bool(None),
            decision: -1,
            assignment_position: state.manage_usize(0),
            reason: None,
            is_implied: state.manage_bool(false),
            hash: rand::random(),
            marked: false,
        }
    }
    
    pub fn set_weight(&mut self, weight: f64) {
        self.weight = Some(weight);
    }
    
    pub fn set_distribution(&mut self, distribution: DistributionIndex) {
        self.distribution = Some(distribution);
    }
    
    pub fn distribution(&self) -> Option<DistributionIndex> {
        self.distribution
    }
    
    pub fn is_probabilitic(&self) -> bool {
        self.weight.is_some()
    }
    
    pub fn weight(&self) -> Option<f64> {
        self.weight
    }
    
    pub fn set_value(&self, value: bool, state: &mut StateManager) {
        state.set_option_bool(self.value, value);
    }
    
    pub fn value(&self, state: &StateManager) -> Option<bool> {
        state.get_option_bool(self.value)
    }
    
    pub fn get_value_index(&self) -> ReversibleOptionBool {
        self.value
    }
    
    pub fn is_fixed(&self, state: &StateManager) -> bool {
        state.get_option_bool(self.value).is_some()
    }
    
    pub fn add_clause_positive_occurence(&mut self, clause: ClauseIndex) {
        self.clauses_positive.push(clause);
    }
    
    pub fn add_clause_negative_occurence(&mut self, clause: ClauseIndex) {
        self.clauses_negative.push(clause);
    }
    
    pub fn iter_clause_positive_occurence(&self) -> impl Iterator<Item = ClauseIndex>  + '_ {
        self.clauses_positive.iter().copied()
    }

    pub fn iter_clause_negative_occurence(&self) -> impl Iterator<Item = ClauseIndex>  + '_ {
        self.clauses_negative.iter().copied()
    }
    
    pub fn set_decision_level(&mut self, level: isize) {
        self.decision = level
    }
    
    pub fn decision_level(&self) -> isize {
        self.decision
    }
    
    pub fn set_reason(&mut self, reason: Option<Reason>, state: &mut StateManager) {
        if reason.is_some() {
            state.set_bool(self.is_implied, true);
        } else {
            state.set_bool(self.is_implied, false);
        }
        self.reason = reason;
    }
    
    pub fn reason(&self, state: &StateManager) -> Option<Reason> {
        if !state.get_bool(self.is_implied) {
            None
        } else {
            self.reason
        }
    }
    
    pub fn hash(&self) -> u64 {
        self.hash
    }
    
    pub fn is_marked(&self) -> bool {
        self.marked
    }
    
    pub fn mark(&mut self) {
        self.marked = true;
    }
    
    pub fn unmark(&mut self) {
        self.marked = false;
    }
    
    pub fn set_assignment_position(&self, position: usize, state: &mut StateManager) {
        state.set_usize(self.assignment_position, position);
    }
    
    pub fn get_assignment_position(&self, state: &StateManager) -> usize {
        state.get_usize(self.assignment_position)
    }
    
    // --- ITERATOR --- //

    pub fn iter_clauses_positive_occurence(&self) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.clauses_positive.iter().copied()
    }
    
    pub fn iter_clauses_negative_occurence(&self) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.clauses_negative.iter().copied()
    }

}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "V{}", self.id + 1)
    }
}

#[cfg(test)]
mod test_variables {
    
    use search_trail::StateManager;

    #[test]
    pub fn create_deterministic() {
    }
}
