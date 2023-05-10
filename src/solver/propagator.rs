//Schlandals
//Copyright (C) 2022 A. Dubray
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
use rug::Float;

#[derive(Debug)]
pub struct Unsat;

pub type PropagationResult = Result<Float, Unsat>;

#[derive(Default)]
pub struct FTReachablePropagator {
    propagation_stack: Vec<(VariableIndex, bool)>,
    pub unconstrained_clauses: Vec<ClauseIndex>,
    t_reachable: Vec<bool>,
    f_reachable: Vec<bool>,
}

impl FTReachablePropagator {
    
    /// Sets the number of clauses for the f-reachable and t-reachable vectors
    pub fn set_number_clauses(&mut self, n: usize) {
        self.t_reachable.resize(n, false);
        self.f_reachable.resize(n, false);
    }
    
    /// Adds a variable to be propagated with the given value
    pub fn add_to_propagation_stack(&mut self, variable: VariableIndex, value: bool) {
        self.propagation_stack.push((variable, value));
    }
    
    /// Propagates a variable to the given value. The component of the variable is also given to be able to use the {f-t}-reachability.
    pub fn propagate_variable(&mut self, variable: VariableIndex, value: bool, g: &mut Graph, state: &mut StateManager, component: ComponentIndex, extractor: &ComponentExtractor) -> PropagationResult {
        self.add_to_propagation_stack(variable, value);
        self.propagate(g, state, component, extractor)
    }
    
    /// Adds a clause to be processed as unconstrained
    pub fn add_unconstrained_clause(&mut self, clause: ClauseIndex, g: &Graph, state: &mut StateManager) {
        if g.is_clause_constrained(clause, state) {
            g.set_clause_unconstrained(clause, state);
            debug_assert!(!self.unconstrained_clauses.contains(&clause));
            self.unconstrained_clauses.push(clause);
        }
    }
    
    /// Computes the unconstrained probability of a distribution. When a distribution does not appear anymore in any constrained
    /// clauses, the probability of branching on it can be pre-computed. This is what this function returns.
    fn get_simplified_distribution_prob(&self, g: &Graph, distribution: DistributionIndex, state: &StateManager) -> Float {
        if g.distribution_number_false(distribution, state) == 0 {
            f128!(1.0)
        } else {
            let mut p = f128!(0.0);
            for w in g.distribution_variable_iter(distribution).filter(|v| !g.is_variable_fixed(*v, state)).map(|v| g.get_variable_weight(v).unwrap()) {
                p += w;
            }
            p
        }
    }
    
    /// Propagates all the unconstrained clauses in the unconstrained clauses stack. It actually updates the sparse-sets of
    /// parents/children in the graph and, if necessary, computes the unconstrained probability of the distributions.
    /// It returns the overall unconstrained probability of the component after the whole propagation.
    pub fn propagate_unconstrained_clauses(&mut self, g: &mut Graph, state: &mut StateManager) -> PropagationResult {
        let mut p = f128!(1.0);
        while let Some(clause) = self.unconstrained_clauses.pop() {
            g.remove_clause_from_children(clause, state);
            g.remove_clause_from_parent(clause, state);
            debug_assert!(!self.unconstrained_clauses.contains(&clause));
            for variable in g.clause_body_iter(clause, state).chain(std::iter::once(g.get_clause_head(clause))) {
                if g.is_variable_probabilistic(variable) && !g.is_variable_fixed(variable, state) {
                    let distribution = g.get_variable_distribution(variable).unwrap();
                    if g.decrement_distribution_constrained_clause_counter(distribution, state) == 0 {
                        p *= self.get_simplified_distribution_prob(g, distribution, state);
                    }
                }
            }
        }
        PropagationResult::Ok(p)
    }
    
    /// Clears the propagation stack as well as the unconstrained clauses stack. This function
    /// is called when an UNSAT has been encountered.
    fn clear(&mut self) {
        self.propagation_stack.clear();
        self.unconstrained_clauses.clear();
    }

    /// Propagates all variables in the propagation stack. The component of being currently solved is also passed as parameter to allow the computation of
    /// the {f-t}-reachability.
    pub fn propagate(&mut self, g: &mut Graph, state: &mut StateManager, component: ComponentIndex, extractor: &ComponentExtractor) -> PropagationResult {
        debug_assert!(self.unconstrained_clauses.is_empty());
        let mut propagation_prob = f128!(1.0);
        while let Some((variable, value)) = self.propagation_stack.pop() {
            if let Some(v) = g.get_variable_value(variable, state) {
                if v == value {
                    continue;
                }
                self.clear();
                return PropagationResult::Err(Unsat);
            }
            g.set_variable(variable, value, state);
            let is_p = g.is_variable_probabilistic(variable);
            for clause in g.variable_clause_body_iter(variable) {
                if g.is_clause_constrained(clause, state) {
                    // The clause is constrained, and var is in its body. If value = T then we need to remove the variable from the body
                    // otherwise the clause is deactivated
                    if value {
                        let head = g.get_clause_head(clause);
                        let head_value = g.get_variable_value(head, state);
                        //debug_assert!(!(head_value.is_some() && head_value.unwrap()));
                        let head_false = head_value.is_some() && !head_value.unwrap();
                        let body_remaining = if is_p {
                            g.clause_decrement_number_probabilistic(clause, state)
                        } else {
                            g.clause_decrement_number_deterministic(clause, state)
                        };
                        if body_remaining == 0 {
                            match head_value {
                                None => self.add_to_propagation_stack(head, true),
                                Some(v) => {
                                    debug_assert!(!v);
                                    self.clear();
                                    return PropagationResult::Err(Unsat);
                                }
                            }
                        } else if body_remaining == 1 && head_false {
                            let v = g.clause_body_iter(clause, state).find(|v| !g.is_variable_fixed(*v, state)).unwrap();
                            self.add_to_propagation_stack(v, false);
                        }
                    } else {
                        self.add_unconstrained_clause(clause, g, state);
                    }
                }
            }
            
            for clause in g.variable_clause_head_iter(variable) {
                if g.is_clause_constrained(clause, state) {
                    if value {
                        self.add_unconstrained_clause(clause, g, state);
                    } else if g.clause_number_unassigned(clause, state) == 1 {
                        let v = g.clause_body_iter(clause, state).find(|v| !g.is_variable_fixed(*v, state)).unwrap();
                        self.propagation_stack.push((v, false));
                    }
                }
            }

            if is_p {
                let distribution = g.get_variable_distribution(variable).unwrap();
                if value {
                    propagation_prob *= g.get_variable_weight(variable).unwrap();
                    if propagation_prob == 0.0 {
                        self.clear();
                        return PropagationResult::Ok(propagation_prob);
                    }
                    for v in g.distribution_variable_iter(distribution).filter(|va| *va != variable) {
                        match g.get_variable_value(v, state) {
                            None => {
                                self.add_to_propagation_stack(v, false)
                            },
                            Some(vv) => {
                                if vv {
                                    self.clear();
                                    return PropagationResult::Err(Unsat);
                                }
                            }
                        };
                    }
                } else if g.distribution_one_left(distribution, state) {
                    if let Some(v) = g.distribution_variable_iter(distribution).find(|v| !g.is_variable_fixed(*v, state)) {
                        self.add_to_propagation_stack(v, true);
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
        propagation_prob *= self.propagate_unconstrained_clauses(g, state)?;
        PropagationResult::Ok(propagation_prob)
    }

    /// Sets the clause to be t-reachable and recursively sets its children to be t-reachable. Notice
    /// that we will never sets a clause that is not in the current components. Since the current components
    /// contains all the constrained clauses reachable from the current clause, it contains all the children of the
    /// clause. Since we juste unit-propagated the components, we only have to check for unconstrained clause to avoid unncessary
    /// computations.
    fn set_t_reachability(&mut self, g: &Graph, state: &StateManager, clause: ClauseIndex) {
        if !self.t_reachable[clause.0] {
            self.t_reachable[clause.0] = true;
            for child in g.children_clause_iter(clause, state) {
                if g.is_clause_constrained(child, state) {
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
            for parent in g.parents_clause_iter(clause, state) {
                if g.is_clause_constrained(parent, state) {
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
            let head = g.get_clause_head(clause);
            let head_value = g.get_variable_value(head, state);
            if (g.is_variable_probabilistic(head) && !g.is_variable_fixed(head, state)) || (head_value.is_some() && !head_value.unwrap()) {
                self.set_f_reachability(g, state, clause);
            }
            if g.clause_number_deterministic(clause, state) == 0 {
                self.set_t_reachability(g, state, clause);
            }
        }
    }

}