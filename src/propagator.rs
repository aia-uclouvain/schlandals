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
//!     3. One of its descendant in the implication graph is f-reachable
//! On the other hand, a clause is t-reachable if
//!     1. Its implicant has no deterministic variable OR
//!     2. One of its ancestor in the implication graph is t-reachable
//! 
//! This is done by a simple traversal of the implication graph, starting from the clauses respecting condition
//! 1,2 for f-reachability or 1 for t-reachability. Finally every unconstrained clause is processed.

use search_trail::StateManager;

use crate::common::f128;
use crate::core::components::{ComponentIndex, ComponentExtractor};
use crate::core::graph::{ClauseIndex, DistributionIndex, Graph, VariableIndex};
use rug::{Assign, Float};

use super::core::literal::Literal;

#[derive(Debug)]
pub struct Unsat;

pub type PropagationResult = Result<(), Unsat>;

pub struct Propagator {
    propagation_stack: Vec<(VariableIndex, bool, Option<ClauseIndex>)>,
    pub unconstrained_clauses: Vec<ClauseIndex>,
    t_reachable: Vec<bool>,
    f_reachable: Vec<bool>,
    assignments: Vec<(DistributionIndex, VariableIndex, bool)>,
    unconstrained_distributions: Vec<DistributionIndex>,
    propagation_prob: Float,
    resolution_stack: Vec<ClauseIndex>,
}

impl Default for Propagator {
    fn default() -> Self {
        Self::new()
    }
}

impl Propagator {
    
    pub fn new() -> Self {
        Self {
            propagation_stack: vec![],
            unconstrained_clauses: vec![],
            t_reachable: vec![],
            f_reachable: vec![],
            assignments: vec![],
            unconstrained_distributions: vec![],
            propagation_prob: f128!(0.0),
            resolution_stack: vec![],
        }
    }
    
    /// Sets the number of clauses for the f-reachable and t-reachable vectors
    pub fn init(&mut self, number_clauses: usize) {
        self.t_reachable.resize(number_clauses, false);
        self.f_reachable.resize(number_clauses, false);
    }
    
    /// Adds a variable to be propagated with the given value
    pub fn add_to_propagation_stack(&mut self, variable: VariableIndex, value: bool, reason: Option<ClauseIndex>) {
        self.propagation_stack.push((variable, value, reason));
    }
    
    /// Propagates a variable to the given value. The component of the variable is also given to be able to use the {f-t}-reachability.
    pub fn propagate_variable(&mut self, variable: VariableIndex, value: bool, g: &mut Graph, state: &mut StateManager, component: ComponentIndex, extractor: &ComponentExtractor, level: isize) -> PropagationResult {
        self.add_to_propagation_stack(variable, value, None);
        self.propagate(g, state, component, extractor, level)
    }
    
    /// Adds a clause to be processed as unconstrained
    pub fn add_unconstrained_clause(&mut self, clause: ClauseIndex, g: &Graph, state: &mut StateManager) {
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
    pub fn assignments_iter(&self) -> impl Iterator<Item = (DistributionIndex, VariableIndex, bool)> + '_{
        self.assignments.iter().copied()
    }

    /// Returns true if there are any assignments in the assignments queue
    pub fn has_assignments(&self) -> bool {
        !self.assignments.is_empty()
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
    fn propagate_unconstrained_distribution(&mut self, g: &Graph, distribution: DistributionIndex, state: &StateManager) {
        if g[distribution].number_false(state) != 0 {
            self.unconstrained_distributions.push(distribution);
            let mut p = f128!(0.0);
            for weight in g[distribution].iter_variables().filter(|v| !g[*v].is_fixed(state)).map(|v| g[v].weight().unwrap()) {
                p += weight;
            }
            self.propagation_prob *= &p;
        }
    }
    
    /// Propagates all the unconstrained clauses in the unconstrained clauses stack. It actually updates the sparse-sets of
    /// parents/children in the graph and, if necessary, computes the unconstrained probability of the distributions.
    /// It returns the overall unconstrained probability of the component after the whole propagation.
    pub fn propagate_unconstrained_clauses(&mut self, g: &mut Graph, state: &mut StateManager) {
        while let Some(clause) = self.unconstrained_clauses.pop() {
            for parent in g[clause].iter_parents(state).collect::<Vec<ClauseIndex>>() {
                g[parent].remove_child(clause, state);
            }
            for child in g[clause].iter_children(state).collect::<Vec<ClauseIndex>>() {
                g[child].remove_parent(clause, state);
            }
            debug_assert!(!self.unconstrained_clauses.contains(&clause));
            for variable in g[clause].iter_variables() {
                if g[variable].is_probabilitic() && !g[variable].is_fixed(state) {
                    let distribution = g[variable].distribution().unwrap();
                    if g[distribution].decrement_constrained(state) == 0 {
                        self.propagate_unconstrained_distribution(g, distribution, state);
                    }
                }
            }
        }
    }
    
    /// Clears the propagation stack as well as the unconstrained clauses stack. This function
    /// is called when an UNSAT has been encountered.
    fn clear(&mut self) {
        self.propagation_stack.clear();
        self.unconstrained_clauses.clear();
        self.assignments.clear();
        self.unconstrained_distributions.clear();
    }

    /// Propagates all variables in the propagation stack. The component of being currently solved is also passed as parameter to allow the computation of
    /// the {f-t}-reachability.
    pub fn propagate(&mut self, g: &mut Graph, state: &mut StateManager, component: ComponentIndex, extractor: &ComponentExtractor, level: isize) -> PropagationResult {
        debug_assert!(self.unconstrained_clauses.is_empty());
        self.assignments.clear();
        self.unconstrained_distributions.clear();
        self.propagation_prob.assign(1.0);
        
        // Find unit clauses
        for clause in extractor.component_iter(component) {
            if g[clause].is_unit(state) {
                let l = g[clause].get_unit_assigment(state);
                self.add_to_propagation_stack(l.to_variable(), l.is_positive(), Some(clause));
            }
        }
        
        while let Some((variable, value, reason)) = self.propagation_stack.pop() {
            if let Some(v) = g[variable].value(state) {
                if v == value {
                    continue;
                }
                self.clear();
                // If reason == None, then the UNSAT is caused by the distribution constraints. For now, no clause learning with that
                if let Some(clause) = reason {
                    //let learned_clause = self.learn_clause_from_conflict(g, state, clause, level);
                    //g.add_clause(learned_clause, None, state, true);
                }
                return PropagationResult::Err(Unsat);
            }
            g.set_variable(variable, value, level, reason, state);
            
            let is_p = g[variable].is_probabilitic();
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
                        self.add_to_propagation_stack(l.to_variable(), if l.is_positive() { true } else { false }, Some(clause));
                    }
                }
            }

            if is_p {
                let distribution = g[variable].distribution().unwrap();
                self.assignments.push((distribution, variable, value));
                if value {
                    self.propagation_prob *= g[variable].weight().unwrap();
                    for v in g[distribution].iter_variables().filter(|va| *va != variable) {
                        match g[v].value(state) {
                            None => {
                                self.add_to_propagation_stack(v, false, None)
                            },
                            Some(vv) => {
                                if vv {
                                    self.clear();
                                    return PropagationResult::Err(Unsat);
                                }
                            }
                        };
                    }
                } else if g[distribution].number_unfixed(state) == 1 {
                    if let Some(v) = g[distribution].iter_variables().find(|v| !g[*v].is_fixed(state)) {
                        self.add_to_propagation_stack(v, true, None);
                    }
                }
            }
        }

        self.set_reachability(g, state, component, extractor);
        for clause in extractor.component_iter(component) {
            if !self.t_reachable[clause.0] || !self.f_reachable[clause.0] {
                self.add_unconstrained_clause(clause, g, state);
            }
        }
        self.propagate_unconstrained_clauses(g, state);
        PropagationResult::Ok(())
    }

    /// Sets the clause to be t-reachable and recursively sets its children to be t-reachable. Notice
    /// that we will never sets a clause that is not in the current components. Since the current components
    /// contains all the constrained clauses reachable from the current clause, it contains all the children of the
    /// clause. Since we juste unit-propagated the components, we only have to check for unconstrained clause to avoid unncessary
    /// computations.
    fn set_t_reachability(&mut self, g: &Graph, state: &StateManager, clause: ClauseIndex) {
        if !self.t_reachable[clause.0] {
            self.t_reachable[clause.0] = true;
            for child in g[clause].iter_children(state) {
                if g[child].is_constrained(state) {
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
    fn set_f_reachability(&mut self, g: &Graph, state: &StateManager, clause: ClauseIndex) {
        if !self.f_reachable[clause.0] {
            self.f_reachable[clause.0] = true;
            for parent in g[clause].iter_parents(state) {
                if g[parent].is_constrained(state) {
                    self.set_f_reachability(g, state, parent);
                }
            }
        }
    }
    
    /// Sets the t-reachability and f-reachability for all clauses in the component
    fn set_reachability(&mut self, g: &Graph, state: &StateManager, component: ComponentIndex, extractor: &ComponentExtractor) {
        self.t_reachable.fill(false);
        self.f_reachable.fill(false);
        for clause in extractor.component_iter(component) {
            if g[clause].is_constrained(state) {
                match g[clause].head() {
                    None => self.set_f_reachability(g, state, clause),
                    Some(h) => {
                        let head = h.to_variable();
                        let head_value = g[head].value(state);
                        if (g[head].is_probabilitic() && !g[head].is_fixed(state)) || (head_value.is_some() && !head_value.unwrap()) {
                            self.set_f_reachability(g, state, clause);
                        }
                    }
                }
                if g[clause].has_probabilistic_in_body(state) {
                    self.set_t_reachability(g, state, clause);
                }
            }
        }
    }
    
    fn learn_clause_from_conflict(&mut self, g: &Graph, state: &mut StateManager, conflict_clause: ClauseIndex, level: isize) -> Vec<Literal> {
        let mut learned_clause: Vec<Literal> = vec![];
        self.resolution_stack.clear();
        
        for variable in g[conflict_clause].iter_variables() {
            learned_clause.push(Literal::from_variable(variable, g[conflict_clause].is_head(variable), g[variable].get_value_index()));
        }
        
        // Apply backwad resolution until only one variable with decision level "level" is in the clause
        loop {
            // First we need to find the literal with which to resolve (i.e. use the clause that forced it to its value
            // for the resolution)
            let mut seen = false;
            let mut i = 0;
            while i < learned_clause.len() {
                let variable = learned_clause[i].to_variable();
                if g[variable].decision_level() == level && g[variable].reason(state).is_some() {
                    if !seen {
                        seen = true;
                    } else {
                        break;
                    }
                }
                i += 1;
            }
            
            if i == learned_clause.len() {
                return learned_clause;
            }
            
            let variable = learned_clause[i].to_variable();
            let reason = g[variable].reason(state).unwrap();
            
            for variable in g[reason].iter_variables() {
                let l = Literal::from_variable(variable, g[reason].is_head(variable), g[variable].get_value_index());
                let mut literal_must_be_added = true;
                for i in 0..learned_clause.len() {
                    if l.to_variable() == learned_clause[i].to_variable() {
                        literal_must_be_added = false;
                        if l.opposite(learned_clause[i]) {
                            learned_clause.swap_remove(i);
                        }
                        break;
                    }
                }
                if literal_must_be_added {
                    learned_clause.push(l);
                }
            }
        }
    }

}

#[cfg(test)]
mod test_clause_learning {
    
    use search_trail::StateManager;
    use crate::core::graph::*;
    use crate::core::components::*;
    use crate::propagator::*;

    fn lit(v: usize, g: &Graph, is_head: bool) -> Literal {
        let variable = VariableIndex(v);
        Literal::from_variable(variable, is_head, g[variable].get_value_index())
    }

   #[test]
   pub fn test_learned_clause() {
        let mut state = StateManager::default();
        let mut g = Graph::new(&mut state, 18, 7);
        let mut propagator = Propagator::new();
        propagator.init(7);
        let extractor = ComponentExtractor::new(&g, &mut state);
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
        
        let r = propagator.propagate_variable(VariableIndex(12), true, &mut g, &mut state, ComponentIndex(0), &extractor, 1);
        assert!(r.is_ok());
        let r = propagator.propagate_variable(VariableIndex(14), true, &mut g, &mut state, ComponentIndex(0), &extractor, 2);
        assert!(r.is_ok());
        let r = propagator.propagate_variable(VariableIndex(16), false, &mut g, &mut state, ComponentIndex(0), &extractor, 3);
        assert!(r.is_ok());
        let r = propagator.propagate_variable(VariableIndex(0), true, &mut g, &mut state, ComponentIndex(0), &extractor, 4);
        assert!(r.is_err());
        //assert_eq!(vec![0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0], propagator.learned_clause);
   } 
}