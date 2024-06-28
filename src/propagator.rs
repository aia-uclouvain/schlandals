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

//! This module give the implentation of the propagator used during the search.
//! It is called at the creation of the solver, to do an initial propagation, and 
//! during the search after all branching decisions.
//! In practice, it first does a boolean unit propagation (BUP) until a fix point is
//! reached.
//! In our case, the BUP works as follows (any inconsistency throws an UNSAT errors and ends
//! the propagation immediatly):
//!     - There is a propagation stack S
//!     - While S is not empty, pop a tuple (variable, value) and assign value to variable
//!     - if value = true:
//!         - If the variable is probabilistic, for all variable v' in the distribution, add (v', false)
//!           to the propagation stack
//!         - Set all clauses which have the variable as their head to be unconstrained
//!         - Remove the variable from the body of all the clauses in which it appears. If the body has no
//!           more variable in it, add to the propagation stack (head, true) with head the head of the clause.
//!     - if value = false:
//!         - If the variable is probabilistic and only one variable (v') remain not fixed in the distribution,
//!           add (v', true) to the propagation stack
//!         - Set all the clauses which have the variable in their implicant as unconstrained
//!         
//! Once this is done, each clause is set as f-reachable or t-reachable. A clause is f-reachable if
//!     1. Its head is set to F OR
//!     2. Its head is an unfixed probabilistic variable OR
//!     3. One of its descendant in the implication problem is f-reachable
//! On the other hand, a clause is t-reachable if
//!     1. Its implicant has no deterministic variable OR
//!     2. One of its ancestor in the implication problem is t-reachable
//! 
//! This is done by a simple traversal of the implication problem, starting from the clauses respecting condition
//! 1,2 for f-reachability or 1 for t-reachability. Finally every unconstrained clause is processed.

use search_trail::{StateManager, UsizeManager, ReversibleUsize};

use crate::common::F128;
use crate::core::components::{ComponentIndex, ComponentExtractor};
use crate::core::problem::{ClauseIndex, DistributionIndex, Problem, VariableIndex};
use rug::{Assign, Float};

use super::core::literal::Literal;
use super::core::variable::Reason;
use super::core::flags::*;

pub type PropagationResult = Result<(), isize>;

pub struct Propagator {
    propagation_stack: Vec<(VariableIndex, bool, isize, Option<Reason>)>,
    pub unconstrained_clauses: Vec<ClauseIndex>,
    clause_flags: Vec<ClauseFlags>,
    lit_flags: Vec<LitFlags>,
    assignments: Vec<Literal>,
    base_assignments: ReversibleUsize,
    unconstrained_distributions: Vec<DistributionIndex>,
    propagation_prob: Float,
    forced: usize,
}

impl Propagator {
    
    pub fn new(state: &mut StateManager) -> Self {
        Self {
            propagation_stack: vec![],
            unconstrained_clauses: vec![],
            clause_flags: vec![],
            lit_flags: vec![],
            assignments: vec![],
            base_assignments: state.manage_usize(0),
            unconstrained_distributions: vec![],
            propagation_prob: F128!(0.0),
            forced: 0,
        }
    }
    
    /// Sets the number of clauses for the f-reachable and t-reachable vectors
    pub fn init(&mut self, number_clauses: usize) {
        self.clause_flags.resize(number_clauses, ClauseFlags::new());
    }
    
    pub fn set_forced(&mut self) {
        self.forced = self.assignments.len();
    }
    
    /// Adds a variable to be propagated with the given value
    pub fn add_to_propagation_stack(&mut self, variable: VariableIndex, value: bool, level: isize, reason: Option<Reason>) {
        self.propagation_stack.push((variable, value, level, reason));
    }
    
    /// Propagates a variable to the given value. The component of the variable is also given to be able to use the {f-t}-reachability.
    pub fn propagate_variable(&mut self, variable: VariableIndex, value: bool, g: &mut Problem, state: &mut StateManager, component: ComponentIndex, extractor: &mut ComponentExtractor, level: isize) -> PropagationResult {
        g[variable].set_reason(None, state);
        self.add_to_propagation_stack(variable, value, level, None);
        self.propagate(g, state, component, extractor, level, false)
    }
    
    /// Adds a clause to be processed as unconstrained
    pub fn add_unconstrained_clause(&mut self, clause: ClauseIndex, g: &Problem, state: &mut StateManager) {
        if g[clause].is_constrained(state) {
            g.set_clause_unconstrained(clause, state);
            debug_assert!(!self.unconstrained_clauses.contains(&clause));
            self.unconstrained_clauses.push(clause);
        }
    }

    /// Returns the propagation probability of the last call to propagate
    pub fn get_propagation_prob(&self) -> &Float {
        &self.propagation_prob
    }
    
    /// Returns an iterator over the assignments made during the last propagation
    pub fn assignments_iter(&self, state: &StateManager) -> impl Iterator<Item = Literal> + '_{
        let start = state.get_usize(self.base_assignments);
        self.assignments.iter().skip(start).copied()
    }

    /// Returns true if there are any assignments in the assignments queue
    pub fn has_assignments(&self, state: &StateManager) -> bool {
        let start = state.get_usize(self.base_assignments);
        start < self.assignments.len()
    }
    
    /// Returns true if there are any unconstrained distributions in the queue
    pub fn has_unconstrained_distribution(&self) -> bool {
        !self.unconstrained_distributions.is_empty()
    }
    
    /// Returns an iterator over the distribution made unconstrained during the last propagation
    pub fn unconstrained_distributions_iter(&self) -> impl Iterator<Item = DistributionIndex> + '_ {
        self.unconstrained_distributions.iter().copied()
    }
    
    /// Computes the unconstrained probability of a distribution. When a distribution does not appear anymore in any constrained
    /// clauses, the probability of branching on it can be pre-computed. This is what this function returns.
    fn propagate_unconstrained_distribution(&mut self, g: &Problem, distribution: DistributionIndex, state: &StateManager) {
        if g[distribution].domain_size(state) == 0 {
            return;
        }
        self.unconstrained_distributions.push(distribution);
        self.propagation_prob *= g[distribution].iter_variables().filter(|v| !g[*v].is_fixed(state)).map(|v| g[v].weight().unwrap()).sum::<f64>();
    }
    
    /// Propagates all the unconstrained clauses in the unconstrained clauses stack. It actually updates the sparse-sets of
    /// parents/children in the problem and, if necessary, computes the unconstrained probability of the distributions.
    /// It returns the overall unconstrained probability of the component after the whole propagation.
    pub fn propagate_unconstrained_clauses(&mut self, g: &mut Problem, state: &mut StateManager) {
        while let Some(clause) = self.unconstrained_clauses.pop() {
            for parent in g[clause].iter_parents(state).collect::<Vec<ClauseIndex>>() {
                g[parent].remove_child(clause, state);
            }
            for child in g[clause].iter_children(state).collect::<Vec<ClauseIndex>>() {
                g[child].remove_parent(clause, state);
            }
            debug_assert!(!self.unconstrained_clauses.contains(&clause));
            for distribution in g[clause].iter_variables().filter(|v| g[*v].is_probabilitic()).map(|v| g[v].distribution().unwrap()).collect::<Vec<DistributionIndex>>() {
                g[distribution].remove_clause(clause, state);
            }
        }
    }
    
    /// Clears the propagation stack as well as the unconstrained clauses stack. This function
    /// is called when an UNSAT has been encountered.
    fn clear(&mut self) {
        self.propagation_stack.clear();
        self.unconstrained_clauses.clear();
        self.unconstrained_distributions.clear();
    }
    
    pub fn restore(&mut self, state: &StateManager) {
        let limit = state.get_usize(self.base_assignments).max(self.forced);
        self.assignments.truncate(limit);
        self.lit_flags.truncate(limit);
        for i in 0..limit {
            self.lit_flags[i].clear();
        }

    }

    /// Propagates all variables in the propagation stack. The component of being currently solved is also passed as parameter to allow the computation of
    /// the {f-t}-reachability.
    pub fn propagate(&mut self, g: &mut Problem, state: &mut StateManager, component: ComponentIndex, extractor: &mut ComponentExtractor, level: isize, is_preproc: bool) -> PropagationResult {
        debug_assert!(self.unconstrained_clauses.is_empty());
        state.set_usize(self.base_assignments, self.assignments.len());
        self.unconstrained_distributions.clear();
        self.propagation_prob.assign(1.0);
        while let Some((variable, value, l, reason)) = self.propagation_stack.pop() {
            if let Some(v) = g[variable].value(state) {
                if v == value {
                    continue;
                }
                self.clear();
                if reason.is_none() {
                    return PropagationResult::Err(level);
                }
                debug_assert!(reason.is_some());
                let (learned_clause, backjump) = self.learn_clause_from_conflict(g, state, reason.unwrap());
                let head = learned_clause.iter().copied().find(|l| l.is_positive());
                let clause = g.add_clause(learned_clause, head, state, true);
                extractor.add_clause_to_component(component, clause);
                return PropagationResult::Err(backjump);
            }
            g[variable].set_assignment_position(self.assignments.len(), state);
            self.assignments.push(Literal::from_variable(variable, value, g[variable].get_value_index()));
            self.lit_flags.push(LitFlags::new());
            g.set_variable(variable, value, l, reason, state);
            
            // TODO: check why this was only g.set_clause_unconstrained and not add unconstrained
            // clause
            if value {
                for clause in g[variable].iter_clauses_positive_occurence(){
                    self.add_unconstrained_clause(clause, g, state);
                }
            } else {
                for clause in g[variable].iter_clauses_negative_occurence(){
                    self.add_unconstrained_clause(clause, g, state);
                }
            }

            let is_p = g[variable].is_probabilitic();

            if !is_preproc {
                for i in (0..g.number_watchers(variable)).rev() {
                    let clause = g.get_clause_watched(variable, i);
                    if g[clause].is_constrained(state) {
                        let new_watcher = g[clause].notify_variable_value(variable, value, is_p, state);
                        if new_watcher != variable {
                            g.remove_watcher(variable, i);
                            g.add_watcher(new_watcher, clause);
                        }
                        if g[clause].is_unit(state) {
                            let l = g[clause].get_unit_assigment(state);
                            self.add_to_propagation_stack(l.to_variable(), l.is_positive(), level, Some(Reason::Clause(clause)));
                        }
                    }
                }
            } else if value {
                for clause in g[variable].iter_clauses_negative_occurence().collect::<Vec<ClauseIndex>>() {
                    g[clause].remove_literals(variable);
                    if g[clause].is_unit(state) {
                        let l = g[clause].get_unit_assigment(state);
                        self.add_to_propagation_stack(l.to_variable(), l.is_positive(), level, Some(Reason::Clause(clause)));
                    }
                }
            } else {
                for clause in g[variable].iter_clauses_positive_occurence().collect::<Vec<ClauseIndex>>() {
                    g[clause].remove_literals(variable);
                    if g[clause].is_unit(state) {
                        let l = g[clause].get_unit_assigment(state);
                        self.add_to_propagation_stack(l.to_variable(), l.is_positive(), level, Some(Reason::Clause(clause)));
                    }
                }
            }

            if is_p {
                let distribution = g[variable].distribution().unwrap();
                if !value {
                    for clause in g[variable].iter_clauses_positive_occurence().collect::<Vec<ClauseIndex>>() {
                        g[distribution].remove_clause(clause, state);
                    }
                }
                if value {
                    g[distribution].set_unconstrained(state);
                    self.propagation_prob *= g[variable].weight().unwrap();
                    for v in g[distribution].iter_variables().filter(|va| !g[*va].is_fixed(state) && *va != variable) {
                        self.add_to_propagation_stack(v, false, level, Some(Reason::Distribution(distribution)));
                    }
                } else if g[distribution].number_unfixed(state) == 1 {
                    g[distribution].set_unconstrained(state);
                    if let Some(v) = g[distribution].iter_variables().find(|v| !g[*v].is_fixed(state)) {
                        self.add_to_propagation_stack(v, true, level, Some(Reason::Distribution(distribution)));
                    }
                }
            }
        }
        self.set_reachability(g, state, component, extractor);
        for clause in extractor.component_iter(component) {
            if !g[clause].is_learned() && !self.clause_flags[clause.0].is_reachable() {
                self.add_unconstrained_clause(clause, g, state);
            }
        }
        self.propagate_unconstrained_clauses(g, state);
        // Possibly the bug: we detect too many unconstrained distribution, some may be constrained
        // by learned clause and have an impact on the problem but we do not detect them because we
        // do not use the learned clause in the branching.
        for distribution in extractor.component_distribution_iter(component) {
            if !g[distribution].is_constrained(state) {
                self.propagate_unconstrained_distribution(g, distribution, state);
            }
        }
        PropagationResult::Ok(())
    }

    /// Sets the clause to be t-reachable and recursively sets its children to be t-reachable. Notice
    /// that we will never sets a clause that is not in the current components. Since the current components
    /// contains all the constrained clauses reachable from the current clause, it contains all the children of the
    /// clause. Since we juste unit-propagated the components, we only have to check for unconstrained clause to avoid unncessary
    /// computations.
    fn set_t_reachability(&mut self, g: &Problem, state: &StateManager, clause: ClauseIndex) {
        if !self.clause_flags[clause.0].is_set(ClauseFlag::TrueReachable) {
            self.clause_flags[clause.0].set(ClauseFlag::TrueReachable);
            for child in g[clause].iter_children(state) {
                if g[child].is_constrained(state) && !g[child].is_learned() {
                    self.set_t_reachability(g, state, child);
                }
            }
        }
    }

    /// Sets the clause to be f-reachable and recursively sets its parents to be f-reachable. Notice
    /// that we will never sets a clause that is not in the current components. Since the current components
    /// contains all the constrained clauses reachable from the current clause, it contains all the parents of the
    /// clause. Since we juste unit-propagated the components, we only have to check for unconstrained clause to avoid unncessary
    /// computations.
    fn set_f_reachability(&mut self, g: &Problem, state: &StateManager, clause: ClauseIndex) {
        if !self.clause_flags[clause.0].is_set(ClauseFlag::FalseReachable) {
            self.clause_flags[clause.0].set(ClauseFlag::FalseReachable);
            for parent in g[clause].iter_parents(state) {
                if g[parent].is_constrained(state) && !g[parent].is_learned() {
                    self.set_f_reachability(g, state, parent);
                }
            }
        }
    }
    
    /// Sets the t-reachability and f-reachability for all clauses in the component
    fn set_reachability(&mut self, g: &mut Problem, state: &mut StateManager, component: ComponentIndex, extractor: &ComponentExtractor) {
        // First we update the parents/child in the problem and clear the flags
        for clause in extractor.component_iter(component){
            if !g[clause].is_learned() {
                self.clause_flags[clause.0].clear();
            }
            for parent in g[clause].iter_parents(state).collect::<Vec<ClauseIndex>>() {
                if !g[parent].is_constrained(state) {
                    g[clause].remove_parent(parent,state);
                    g[parent].remove_child(clause, state);
                }
            }

            for child in g[clause].iter_children(state).collect::<Vec<ClauseIndex>>() {
                if !g[child].is_constrained(state) {
                    g[clause].remove_child(child,state);
                    g[child].remove_parent(clause, state);
                }
            }
        }


        for clause in extractor.component_iter(component){
            if g[clause].is_constrained(state) && !g[clause].is_learned() {
                debug_assert!(!g[clause].is_unit(state));
                match g[clause].head() {
                    None => self.set_f_reachability(g, state, clause),
                    Some(h) => {
                        let head = h.to_variable();
                        let head_value = g[head].value(state);
                        match head_value {
                            None => {
                                if g[head].is_probabilitic() {
                                    self.set_f_reachability(g, state, clause);
                                }
                            },
                            Some(b) => {
                                if !b {
                                    self.set_f_reachability(g, state, clause);
                                }
                            }
                        }
                    }
                }
                if !g[clause].has_deterministic_in_body(state) {
                    self.set_t_reachability(g, state, clause);
                }
            }
        }
    }
    
    fn is_uip(&self, cursor: usize, g: &Problem, state: &StateManager) -> bool {
        let variable = self.assignments[cursor].to_variable();
        let assignment_pos = g[variable].get_assignment_position(state);
        if g[variable].reason(state).is_none() {
            return true;
        }
        if !self.lit_flags[assignment_pos].is_set(LitFlag::IsMarked) {
            return false;
        }
        for i in (self.forced..cursor).rev() {
            let lit = self.assignments[i];
            if self.lit_flags[i].is_set(LitFlag::IsMarked) {
                return false;
            }
            if g[lit.to_variable()].reason(state).is_none() {
                return true;
            }
        }
        return false;
    }
    
    fn is_implied(&mut self, lit: Literal, g: &Problem, state: &StateManager) -> bool {
        let pos = g[lit.to_variable()].get_assignment_position(state);
        if self.lit_flags[pos].is_set(LitFlag::IsImplied){
            return true;
        }
        if self.lit_flags[pos].is_set(LitFlag::IsNotImplied) {
            return false;
        }
        
        match g[lit.to_variable()].reason(state) {
            None => return false,
            Some(r) => {
                match r {
                    Reason::Clause(c) => {
                        for p in g[c].iter_variables().map(|v| g[v].get_assignment_position(state)).filter(|p| *p != pos) {
                            let l = self.assignments[p];
                            if !self.lit_flags[p].is_set(LitFlag::IsMarked) && !self.is_implied(l, g, state) {
                                self.lit_flags[pos].set(LitFlag::IsNotImplied);
                                return false;
                            }
                        }
                        self.lit_flags[pos].set(LitFlag::IsImplied);
                        return true;
                    },
                    Reason::Distribution(d) => {
                        if lit.is_positive() {
                            for p in g[d].iter_variables().map(|v| g[v].get_assignment_position(state)).filter(|p| *p != pos) {
                                let l = self.assignments[p];
                                if !self.lit_flags[p].is_set(LitFlag::IsMarked) && !self.is_implied(l, g, state) {
                                    self.lit_flags[pos].set(LitFlag::IsNotImplied);
                                    return false;
                                }
                            }
                            self.lit_flags[pos].set(LitFlag::IsImplied);
                            return true;
                        } else {
                            let assigned = g[d].iter_variables().find(|v| g[*v].value(state).is_some() && g[*v].value(state).unwrap()).unwrap();
                            let p = g[assigned].get_assignment_position(state);
                            let l = self.assignments[p];
                            if !self.lit_flags[p].is_set(LitFlag::IsMarked) && !self.is_implied(l, g, state) {
                                self.lit_flags[pos].set(LitFlag::IsNotImplied);
                                return false;
                            }
                            self.lit_flags[pos].set(LitFlag::IsImplied);
                            return true;
                        }
                    }
                }
            }
        }
    }
    
    fn learn_clause_from_conflict(&mut self, g: &mut Problem, state: &mut StateManager, conflict_clause: Reason) -> (Vec<Literal>, isize) {
        match conflict_clause {
            Reason::Clause(c) => {
                for variable in g[c].iter_variables() {
                    self.lit_flags[g[variable].get_assignment_position(state)].set(LitFlag::IsMarked);
                }
            },
            Reason::Distribution(d) => {
                let number_true = g[d].iter_variables().filter(|v| g[*v].is_fixed(state) && g[*v].value(state).unwrap()).count();
                if number_true > 1 {
                    for variable in g[d].iter_variables().filter(|v| g[*v].is_fixed(state) && g[*v].value(state).unwrap()) {
                        self.lit_flags[g[variable].get_assignment_position(state)].set(LitFlag::IsMarked);
                    }
                } else {
                    debug_assert!(number_true == 0);
                    for variable in g[d].iter_variables() {
                        self.lit_flags[g[variable].get_assignment_position(state)].set(LitFlag::IsMarked);
                    }
                }
            }
        };
        let mut cursor = self.assignments.len();
        loop {
            cursor -= 1;
            if cursor < self.forced {
                break;
            }
            
            // Check if the current assignment is an UIP
            let lit = self.assignments[cursor];
            let variable = lit.to_variable();
            let v_pos = g[variable].get_assignment_position(state);
            
            if self.is_uip(cursor, g, state) {
                break;
            }
            
            if !self.lit_flags[v_pos].is_set(LitFlag::IsMarked){
                continue;
            }
            
            match g[variable].reason(state).unwrap() {
                Reason::Clause(clause) => {
                    for pos in g[clause].iter_variables().map(|v| g[v].get_assignment_position(state)) {
                        self.lit_flags[pos].set(LitFlag::IsMarked);
                    }
                },
                Reason::Distribution(distribution) => {
                    let number_true = g[distribution].iter_variables().filter(|v| g[*v].is_fixed(state) && g[*v].value(state).unwrap()).count();
                    if number_true > 1 {
                        for variable in g[distribution].iter_variables().filter(|v| g[*v].is_fixed(state) && g[*v].value(state).unwrap()) {
                            self.lit_flags[g[variable].get_assignment_position(state)].set(LitFlag::IsMarked);
                        }
                    } else {
                        for variable in g[distribution].iter_variables() {
                            self.lit_flags[g[variable].get_assignment_position(state)].set(LitFlag::IsMarked);
                        }
                    }
                }
            };
        }
        
        
        let mut learned: Vec<Literal> = vec![];
        // We build the clause from based on the UIP
        for i in (self.forced..cursor+1).rev() {
            let lit = self.assignments[i];
            if self.lit_flags[i].is_set(LitFlag::IsMarked) && !self.is_implied(lit, g, state) {
                learned.push(lit.opposite());
            }
        }
        
        let mut count_used = 0;
        let mut backjump = cursor;
        for i in (self.forced..cursor+1).rev() {
            let lit = self.assignments[i];
            if self.lit_flags[i].is_set(LitFlag::IsInConflictClause) {
                count_used += 1;
            }
            if count_used == 1 && g[lit.to_variable()].reason(state).is_none() {
                backjump = i;
            }
        }
        
        (learned, g[self.assignments[backjump].to_variable()].decision_level() as isize)
    }

    pub fn iter_propagated_assignments(&self) -> impl Iterator<Item = Literal> + '_ {
        self.assignments.iter().copied()
    }

    pub fn reduce(&mut self, number_clauses: usize, number_variables: usize) {
        self.clause_flags.truncate(number_clauses);
        self.clause_flags.shrink_to_fit();
        self.lit_flags.truncate(number_variables);
        self.lit_flags.shrink_to_fit();
        self.assignments.clear();
    }
}

#[cfg(test)]
mod test_clause_learning {
    
    use search_trail::StateManager;
    use crate::core::problem::*;
    use crate::core::components::*;
    use crate::propagator::*;

    fn lit(v: usize, g: &Problem, is_head: bool) -> Literal {
        let variable = VariableIndex(v);
        Literal::from_variable(variable, is_head, g[variable].get_value_index())
    }

   #[test]
   pub fn test_learned_clause() {
        let mut state = StateManager::default();
        let mut g = Problem::new(&mut state, 18, 7);
        let mut propagator = Propagator::new(&mut state);
        propagator.init(7);
        let mut extractor = ComponentExtractor::new(&g, &mut state);
        g.add_distributions(&vec![
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ], &mut state);

        g.add_clause(vec![
            lit(2, &g, true),
            lit(0, &g, false),
        ], Some(lit(2, &g, true)), &mut state, false);
        g.add_clause(vec![
            lit(4, &g, true),
            lit(0, &g, false),
            lit(12, &g, false),
        ], Some(lit(4, &g, true)), &mut state, false);
        g.add_clause(vec![
            lit(6, &g, true),
            lit(2, &g, false),
            lit(4, &g, false),
        ], Some(lit(6, &g, true)), &mut state, false);
        g.add_clause(vec![
            lit(16, &g, true),
            lit(6, &g, false),
            lit(10, &g, false),
        ], Some(lit(16, &g, true)), &mut state, false);
        g.add_clause(vec![
            lit(8, &g, true),
            lit(14, &g, false),
            lit(6, &g, false),
        ], Some(lit(8, &g, true)), &mut state, false);
        g.add_clause(vec![
            lit(10, &g, true),
            lit(8, &g, false),
        ], Some(lit(10, &g, true)), &mut state, false);
        
        let r = propagator.propagate_variable(VariableIndex(12), true, &mut g, &mut state, ComponentIndex(0), &mut extractor, 1);
        assert!(r.is_ok());
        let r = propagator.propagate_variable(VariableIndex(14), true, &mut g, &mut state, ComponentIndex(0), &mut extractor, 2);
        assert!(r.is_ok());
        let r = propagator.propagate_variable(VariableIndex(16), false, &mut g, &mut state, ComponentIndex(0), &mut extractor, 3);
        assert!(r.is_ok());
        //let r = propagator.propagate_variable(VariableIndex(0), true, &mut g, &mut state, ComponentIndex(0), &mut extractor, 4);
        //assert!(r.is_err());
        //assert_eq!(vec![0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0], propagator.learned_clause);
   } 
}
