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

use crate::core::components::{ComponentIndex, ComponentExtractor};
use crate::core::problem::{ClauseIndex, Problem, VariableIndex};

use super::core::literal::Literal;
use super::core::flags::*;

pub type PropagationResult = Result<(), isize>;

pub struct Propagator {
    propagation_stack: Vec<(VariableIndex, bool, isize)>,
    clause_flags: Vec<ClauseFlags>,
    assignments: Vec<Literal>,
    base_assignments: ReversibleUsize,
}

impl Propagator {
    
    pub fn new(state: &mut StateManager) -> Self {
        Self {
            propagation_stack: vec![],
            clause_flags: vec![],
            assignments: vec![],
            base_assignments: state.manage_usize(0),
        }
    }
    
    /// Sets the number of clauses for the f-reachable and t-reachable vectors
    pub fn init(&mut self, number_clauses: usize) {
        self.clause_flags.resize(number_clauses, ClauseFlags::new());
    }
    
    /// Adds a variable to be propagated with the given value
    pub fn add_to_propagation_stack(&mut self, variable: VariableIndex, value: bool, level: isize) {
        self.propagation_stack.push((variable, value, level));
    }
    
    /// Propagates a variable to the given value. The component of the variable is also given to be able to use the {f-t}-reachability.
    pub fn propagate_variable(&mut self, variable: VariableIndex, value: bool, g: &mut Problem, state: &mut StateManager, component: ComponentIndex, extractor: &mut ComponentExtractor, level: isize) -> PropagationResult {
        self.add_to_propagation_stack(variable, value, level);
        self.propagate(g, state, component, extractor, level)
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
    
    /// Clears the propagation stack as well as the unconstrained clauses stack. This function
    /// is called when an UNSAT has been encountered.
    fn clear(&mut self) {
        self.propagation_stack.clear();
    }
    
    pub fn restore(&mut self, state: &StateManager) {
        let limit = state.get_usize(self.base_assignments);
        self.assignments.truncate(limit);
    }

    /// Propagates all variables in the propagation stack. The component of being currently solved is also passed as parameter to allow the computation of
    /// the {f-t}-reachability.
    pub fn propagate(&mut self, g: &mut Problem, state: &mut StateManager, component: ComponentIndex, extractor: &mut ComponentExtractor, level: isize) -> PropagationResult {
        state.set_usize(self.base_assignments, self.assignments.len());
        while let Some((variable, value, l)) = self.propagation_stack.pop() {
            if let Some(v) = g[variable].value(state) {
                if v == value {
                    continue;
                }
                self.clear();
                return PropagationResult::Err(level);
            }
            g[variable].set_assignment_position(self.assignments.len(), state);
            self.assignments.push(Literal::from_variable(variable, value, g[variable].get_value_index()));
            g.set_variable(variable, value, l, state);
            
            if value {
                for clause in g[variable].iter_clauses_positive_occurence(state){
                    g[clause].deactivate(state);
                }
            } else {
                for clause in g[variable].iter_clauses_negative_occurence(state){
                    g[clause].deactivate(state);
                }
                for clause in g[variable].iter_clauses_positive_occurence(state){
                    g[clause].set_head_f_reachable(state);
                }
            }

            for i in (0..g.number_watchers(variable)).rev() {
                let clause = g.get_clause_watched(variable, i);
                if g[clause].is_active(state) {
                    g[clause].modified(state);
                    let new_watcher = g[clause].notify_variable_value(variable, state);
                    if new_watcher != variable {
                        g.remove_watcher(variable, i);
                        g.add_watcher(new_watcher, clause);
                    }
                    if g[clause].is_unit(state) {
                        let l = g[clause].get_unit_assigment(state);
                        self.add_to_propagation_stack(l.to_variable(), l.is_positive(), level);
                    }
                }
            }

            if g[variable].is_probabilitic() {
                let distribution = g[variable].distribution().unwrap();
                if value {
                    for v in g[distribution].iter_variables().filter(|va| !g[*va].is_fixed(state) && *va != variable) {
                        self.add_to_propagation_stack(v, false, level);
                    }
                } else if g[distribution].size(state) == 1 {
                    if let Some(v) = g[distribution].iter_variables().find(|v| !g[*v].is_fixed(state)) {
                        self.add_to_propagation_stack(v, true, level);
                    }
                }
            } else if value {
                for clause in g[variable].iter_clauses_negative_occurence(state) {
                    g[clause].decrement_deterministic_in_body(state);
                }
            }
        }
        self.set_reachability(g, state, component, extractor);
        for clause in extractor.component_iter(component) {
            if !g[clause].is_learned() && !self.clause_flags[clause.0].is_reachable() {
                g.deactivate_clause(clause, state);
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
                if !g[child].is_learned() && g[child].is_active(state) {
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
                if !g[parent].is_learned() && g[parent].is_active(state) {
                    self.set_f_reachability(g, state, parent);
                }
            }
        }
    }
    
    /// Sets the t-reachability and f-reachability for all clauses in the component
    fn set_reachability(&mut self, g: &mut Problem, state: &mut StateManager, component: ComponentIndex, extractor: &ComponentExtractor) {
        // First we update the parents/child in the problem and clear the flags
        for clause in extractor.component_iter(component){
            if g[clause].is_learned() {
                continue;
            }
            self.clause_flags[clause.0].clear();
            for parent_index in (0..g[clause].number_constrained_parents(state)).rev() {
                let parent = g[clause].get_parent_at(parent_index);
                if !g[parent].is_active(state) {
                    g[clause].remove_parent_at(parent_index, state);
                }
            }

            for child_index in (0..g[clause].number_constrained_children(state)).rev() {
                let child = g[clause].get_child_at(child_index);
                if !g[child].is_active(state) {
                    g[clause].remove_child_at(child_index, state);
                }
            }
        }

        for clause in extractor.component_iter(component){
            if !g[clause].is_learned() && g[clause].is_active(state) {
                if g[clause].is_head_f_reachable(state) {
                    self.set_f_reachability(g, state, clause);
                }
                if !g[clause].has_deterministic_in_body(state) {
                    self.set_t_reachability(g, state, clause);
                }
            }
        }
    }
    
    pub fn reduce(&mut self, number_clauses: usize, number_variables: usize, state: &mut StateManager) {
        self.clause_flags.truncate(number_clauses);
        self.clause_flags.shrink_to_fit();
        self.assignments.clear();
        state.set_usize(self.base_assignments, 0);
    }
}
