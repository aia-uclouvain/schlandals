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
use search_trail::StateManager;

use crate::common::f128;
use crate::core::graph::{ClauseIndex, DistributionIndex, Graph, VariableIndex};
use rug::Float;

#[derive(Debug)]
pub struct Unsat;

pub type PropagationResult = Result<Float, Unsat>;

#[derive(Default)]
pub struct FTReachablePropagator {
    propagation_stack: Vec<(VariableIndex, bool)>,
    unconstrained_clauses: Vec<ClauseIndex>,
}

impl FTReachablePropagator {
    
    pub fn add_to_propagation_stack(&mut self, variable: VariableIndex, value: bool) {
        self.propagation_stack.push((variable, value));
    }
    
    pub fn propagate_variable(&mut self, variable: VariableIndex, value: bool, g: &mut Graph, state: &mut StateManager) -> PropagationResult {
        self.add_to_propagation_stack(variable, value);
        self.propagate(g, state)
    }
    
    fn add_unconstrained_clause(&mut self, clause: ClauseIndex, g: &Graph, state: &mut StateManager) {
        if g.is_clause_constrained(clause, state) {
            g.set_clause_unconstrained(clause, state);
            self.unconstrained_clauses.push(clause);
        }
    }
    
    pub fn propagate_unreachable_clauses(&mut self, g: &mut Graph, state: &mut StateManager) -> PropagationResult {
        for clause in g.clause_iter() {
            if !g.is_clause_f_reachable(clause, state) || !g.is_clause_t_reachable(clause, state) {
                self.add_unconstrained_clause(clause, g, state);
            }
        }
        self.propagate(g, state)
    }
    
    fn get_simplified_distribution_prob(&self, g: &Graph, distribution: DistributionIndex, state: &StateManager) -> Float {
        let mut p = f128!(0.0);
        for w in g.distribution_variable_iter(distribution).filter(|v| !g.is_variable_bound(*v, state)).map(|v| g.get_variable_weight(v).unwrap()) {
            p += w;
        }
        p
    }
    
    fn deactivate_clauses(&mut self, g: &mut Graph, state: &mut StateManager) -> PropagationResult {
        let mut p = f128!(1.0);
        while let Some(clause) = self.unconstrained_clauses.pop() {
            for variable in g.clause_body_iter(clause) {
                if g.is_variable_probabilistic(variable) && !g.is_variable_bound(variable, state) {
                    let distribution = g.get_variable_distribution(variable).unwrap();
                    if g.decrement_distribution_clause_counter(distribution, state) == 0 {
                        p *= self.get_simplified_distribution_prob(g, distribution, state);
                    }
                }
            }
            // The clause is no more f-reachable, need to update the parents if necessary
            // Since the iterator on the edges borrow the graph, we must collect it into a vec.
            // In practice this update of t and f reachability should not happen at every node of the search, so the cost is
            // limited
            for edge in g.parents_clause_iter(clause) {
                let parent = g.get_edge_source(edge);
                let head = g.get_clause_head(parent);
                let head_value = g.get_variable_value(head, state);
                let head_false = head_value.is_some() && !head_value.unwrap();
                if !head_false && g.is_clause_constrained(parent, state) && g.clause_decrement_active_children(parent, state) == 0 {
                    self.add_unconstrained_clause(parent, g, state);
                }
            }
            // Same for t-reachability
            for edge in g.children_clause_iter(clause) {
                let child = g.get_edge_destination(edge);
                if g.is_clause_constrained(child, state) && g.clause_decrement_active_parents(child, state) == 0 && g.clause_number_deterministic(child, state) != 0 {
                    self.add_unconstrained_clause(child, g, state);
                }
            }
        }
        PropagationResult::Ok(p)
    }

    pub fn propagate(&mut self, g: &mut Graph, state: &mut StateManager) -> PropagationResult {
        let mut propagation_prob = f128!(1.0);
        while let Some((variable, value)) = self.propagation_stack.pop() {
            if let Some(v) = g.get_variable_value(variable, state) {
                if v == value {
                    continue;
                }
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
                                    return PropagationResult::Err(Unsat);
                                }
                            }
                        } else if body_remaining == 1 && head_false {
                            let v = g.clause_body_iter(clause).find(|v| !g.is_variable_bound(*v, state)).unwrap();
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
                        let v = g.clause_body_iter(clause).find(|v| !g.is_variable_bound(*v, state)).unwrap();
                        self.propagation_stack.push((v, false));
                    }
                }
            }
            
            if is_p {
                let distribution = g.get_variable_distribution(variable).unwrap();
                if value {
                    propagation_prob *= g.get_variable_weight(variable).unwrap();
                    for v in g.distribution_variable_iter(distribution).filter(|va| *va != variable) {
                        // TODO: opti, check if variable is bound before adding to propagation stack
                        self.add_to_propagation_stack(v, false);
                    }
                } else if g.distribution_one_left(distribution, state) {
                    if let Some(v) = g.distribution_variable_iter(distribution).find(|v| !g.is_variable_bound(*v, state)) {
                        self.add_to_propagation_stack(v, true);
                    }
                }
            }
        }
        propagation_prob *= self.deactivate_clauses(g, state)?;
        PropagationResult::Ok(propagation_prob)
    }

    /*
    pub fn propagate(&mut self, g: &mut Graph, state: &mut StateManager) -> PropagationResult {
        while !self.propagation_stack.is_empty() {
            g.set_variable(variable, value, state);
            let is_p = g.is_variable_probabilistic(variable);
            // For probabilistic variables add the probability to the propagation prob.
            // If the probability goes down to 0 then it can never be higher than that, return
            // Actually do the propagations.
            // If value is T then
            //      - Remove the variable from the body of every active clause in which it appears
            //      - Deactivate all the clauses in which it is the head
            // If value is F then
            //      - Deactive all the clauses in which it is in the body
            //      - For all clause in which it is the head, check if the body is of length 1.
            //        If so, set the remaining var to F
            if value {
                for clause in g.variable_clause_body_iter(variable) {
                    g.clause_remove_variable_hash(clause, variable, state);
                    let head = g.get_clause_head(clause);
                    let head_false = g.is_variable_bound(head, state) && !g.get_variable_value(head);
                    let head_true = g.is_variable_bound(variable, state) && g.get_variable_value(head);
                    let body_remaining = if is_p {
                        let distribution = g.get_variable_distribution(variable).unwrap();
                        if g.decrement_distribution_clause_counter(distribution, state) == 0 {
                            propagation_prob *= self.get_simplified_distribution_prob(g, distribution, state);
                            if propagation_prob == 0.0 {
                                return PropagationResult::Ok(propagation_prob);
                            }
                        }
                        g.clause_decrement_number_probabilistic(clause, state)
                    } else {
                        g.clause_decrement_number_deterministic(clause, state)
                    };
                    if body_remaining == 0 && head_true {
                        g.deactivate_clause(clause, state);
                    } else if body_remaining == 1 && head_false {
                        let v = g.clause_body_iter(clause).filter(|v| !g.is_variable_bound(*v, state)).next().unwrap();
                        self.propagation_stack.push((v, false));
                        g.deactivate_clause(clause, state);
                    } else if body_remaining == 0 {
                        self.propagation_stack.push((head, true));
                        g.deactivate_clause(clause, state);
                    }
                }
                
                for clause in g.variable_clause_head_iter(variable).collect::<Vec<ClauseIndex>>() {
                    propagation_prob *= self.deactivate_clause(g, clause, state)?;
                }
            } else {
                for clause in g.variable_clause_head_iter(variable) {
                    if !g.is_clause_f_reachable(clause, state) {
                        g.set_clause_bot_reachable(clause, state);
                    }
                    if g.clause_number_unassigned(clause, state) == 1 {
                        let v = g.clause_body_iter(clause).filter(|v| !g.is_variable_bound(*v, state)).next().unwrap();
                        self.propagation_stack.push((v, false));
                    }
                }
                for clause in g.variable_clause_body_iter(variable).collect::<Vec<ClauseIndex>>() {
                    propagation_prob *= self.deactivate_clause(g, clause, state)?;
                }
            }
            if is_p {
                let distribution = g.get_variable_distribution(variable).unwrap();
                if value {
                    propagation_prob *= g.get_variable_weight(variable).unwrap();
                    if propagation_prob == 0.0 {
                        return PropagationResult::Ok(propagation_prob);
                    }
                    for other in g.distribution_variable_iter(distribution).filter(|v| *v != variable) {
                        if !g.is_variable_bound(other, state) {
                            self.add_to_propagation_stack(other, false);
                        } else if g.get_variable_value(other) {
                            return PropagationResult::Err(Unsat);
                        }
                    }
                } else {
                    if g.distribution_one_left(distribution, state) {
                        match g.distribution_variable_iter(distribution).filter(|v| !g.is_variable_bound(*v, state)).next() {
                            Some(v) => self.add_to_propagation_stack(v, true),
                            None => (),
                        }
                    }
                }
            }
        }
        PropagationResult::Ok(propagation_prob)
    }
    */
}

/*
#[cfg(test)]
mod test_simple_propagator_propagation {

    use crate::core::graph::{Graph, NodeIndex};
    use crate::core::trail::StateManager;
    use crate::solver::propagator::SimplePropagator;

    #[test]
    fn initial_propagation_simple_implications() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        let d: Vec<NodeIndex> = (0..4)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect();
        let p: Vec<NodeIndex> = (0..4)
            .map(|_| g.add_distribution(&vec![0.1], &mut state)[0])
            .collect();

        // determinisdeativae_lause
        // d[1] -> d[0]
        g.add_clause(d[0], &vec![d[1]], &mut state);
        // deterministic -> probabilistic
        // d[2] -> p[0]
        g.add_clause(p[0], &vec![d[2]], &mut state);
        // probabilistic -> deterministic
        // p[1] -> d[3]
        g.add_clause(d[3], &vec![p[1]], &mut state);
        // probabilistic -> probabilistic
        // p[2] -> p[3]
        g.add_clause(p[3], &vec![p[2]], &mut state);

        g.propagate(&mut state).unwrap();
        assert!(g.is_node_bound(d[0], &state));
        assert!(g.is_node_bound(d[1], &state));
        assert!(g.is_node_bound(d[2], &state));
        assert!(g.is_node_bound(d[3], &state));
        assert_eq!(true, g.get_node_value(d[0]));
        assert_eq!(true, g.get_node_value(d[1]));
        assert_eq!(false, g.get_node_value(d[2]));
        assert_eq!(true, g.get_node_value(d[3]));
    }

    #[test]
    fn initial_propagation_chained_implications() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        let d: Vec<NodeIndex> = (0..6)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect();
        let p: Vec<NodeIndex> = (0..3)
            .map(|_| g.add_distribution(&vec![0.1], &mut state)[0])
            .collect();

        // d[0] -> d[1] -> d[2]
        g.add_clause(d[1], &vec![d[0]], &mut state);
        g.add_clause(d[2], &vec![d[1]], &mut state);

        // d[3] -> p[0] -> d[4]
        g.add_clause(p[0], &vec![d[3]], &mut state);
        g.add_clause(d[4], &vec![p[0]], &mut state);

        // p[1] -> d[5] -> p[2]
        g.add_clause(d[5], &vec![p[1]], &mut state);
        g.add_clause(p[2], &vec![d[5]], &mut state);

        g.propagate(&mut state).unwrap();

        assert!(g.is_node_bound(d[0], &state));
        assert!(g.is_node_bound(d[1], &state));
        assert!(g.is_node_bound(d[2], &state));
        assert!(g.is_node_bound(d[3], &state));
        assert!(g.is_node_bound(d[4], &state));
        assert!(!g.is_node_bound(d[5], &state));
        assert!(!g.is_node_bound(p[1], &state));
        assert!(!g.is_node_bound(p[2], &state));

        assert_eq!(false, g.get_node_value(d[0]));
        assert_eq!(false, g.get_node_value(d[1]));
        assert_eq!(false, g.get_node_value(d[2]));
        assert_eq!(false, g.get_node_value(d[3]));
        assert_eq!(true, g.get_node_value(d[4]));
    }
}

#[cfg(test)]
mod test_simple_propagator_node_propagation {
    use crate::core::graph::{Graph, NodeIndex};
    use crate::core::trail::{SaveAndRestore, StateManager};
    use crate::solver::propagator::SimplePropagator;

    //#[test]
    fn simple_implications() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        let d = g.add_node(false, None, None, &mut state);
        let p1 = g.add_distribution(&vec![1.0], &mut state);
        let p2 = g.add_distribution(&vec![1.0], &mut state);
        let d2 = g.add_node(false, None, None, &mut state);
        let d3 = g.add_node(false, None, None, &mut state);

        // d3 -> p1
        // p1 -> d
        // d -> p2
        // p2 -> d2
        g.add_clause(p1[0], &vec![d3], &mut state);
        g.add_clause(d, &vec![p1[0]], &mut state);
        g.add_clause(p2[0], &vec![d], &mut state);
        g.add_clause(d2, &vec![p2[0]], &mut state);

        state.save_state();

        match g.propagate_node(p1[0], true, &mut state) {
            Ok(_) => (),
            Err(_) => (),
        };

        assert!(g.is_node_bound(d, &state));
        assert!(g.is_node_bound(p1[0], &state));
        assert!(g.is_node_bound(p2[0], &state));
        assert_eq!(true, g.get_node_value(d));
        assert_eq!(true, g.get_node_value(p1[0]));
        assert_eq!(true, g.get_node_value(p2[0]));

        state.restore_state();
        state.save_state();

        match g.propagate_node(p1[0], false, &mut state) {
            Ok(_) => (),
            Err(_) => (),
        };
        // This assert is not valid anymore since component with only distribution are computed during the propagation
        //assert!(g.is_node_bound(d, &state));
        assert!(g.is_node_bound(p1[0], &state));
        assert!(!g.is_node_bound(p2[0], &state));
        assert_eq!(false, g.get_node_value(p1[0]));

        state.restore_state();
        state.save_state();

        g.propagate_node(p2[0], true, &mut state).unwrap();
        assert!(g.is_node_bound(d, &state));
        assert!(!g.is_node_bound(p1[0], &state));
        assert!(g.is_node_bound(p2[0], &state));
        assert_eq!(true, g.get_node_value(d));
        assert_eq!(true, g.get_node_value(p2[0]));

        state.restore_state();
        state.save_state();

        match g.propagate_node(p2[0], false, &mut state) {
            Ok(_) => (),
            Err(_) => (),
        };
        assert!(g.is_node_bound(d, &state));
        assert!(g.is_node_bound(p1[0], &state));
        assert!(g.is_node_bound(p2[0], &state));
        assert_eq!(false, g.get_node_value(d));
        assert_eq!(false, g.get_node_value(p1[0]));
        assert_eq!(false, g.get_node_value(p2[0]));
    }

    //#[test]
    fn test_multiple_edges_different_clauses() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        let d = g.add_node(false, None, None, &mut state);
        let d2 = g.add_node(false, None, None, &mut state);
        let d3 = g.add_node(false, None, None, &mut state);
        let d4 = g.add_node(false, None, None, &mut state);
        let d5 = g.add_node(false, None, None, &mut state);
        let dp1 = g.add_distribution(&vec![0.5, 0.5], &mut state);
        let dp2 = g.add_distribution(&vec![0.5, 0.5], &mut state);
        let p1 = dp1[0];
        let p2 = dp1[1];
        let p3 = dp2[0];
        let p4 = dp2[1];

        //
        // d2 -> p1 -> d ----> p3 -> d4
        // d3 -> p2 ---|  |
        //                |--> p4 -> d5

        g.add_clause(p1, &vec![d2], &mut state);
        g.add_clause(p2, &vec![d3], &mut state);
        g.add_clause(d4, &vec![p3], &mut state);
        g.add_clause(d5, &vec![p4], &mut state);

        g.add_clause(d, &vec![p1], &mut state);
        g.add_clause(d, &vec![p2], &mut state);
        g.add_clause(p3, &vec![d], &mut state);
        g.add_clause(p4, &vec![d], &mut state);

        state.save_state();

        match g.propagate_node(p1, true, &mut state) {
            Ok(_) => (),
            Err(_) => (),
        };
        assert!(g.is_node_bound(d, &state));
        assert!(g.is_node_bound(p1, &state));
        assert!(!g.is_node_bound(p2, &state));
        assert!(g.is_node_bound(p3, &state));
        assert!(g.is_node_bound(p4, &state));
        assert_eq!(true, g.get_node_value(d));
        assert_eq!(true, g.get_node_value(p1));
        assert_eq!(false, g.get_node_value(p3));
        assert_eq!(true, g.get_node_value(p4));

        state.restore_state();
        state.save_state();

        match g.propagate_node(p1, false, &mut state) {
            Ok(_) => (),
            Err(_) => (),
        };
        assert!(g.is_node_bound(d, &state));
        assert_eq!(true, g.get_node_value(d));
        assert!(g.is_node_bound(p1, &state));
        assert!(g.is_node_bound(p2, &state));
        assert_eq!(true, g.get_node_value(p2));
        assert!(!g.is_node_bound(p3, &state));
        assert!(!g.is_node_bound(p4, &state));
        assert_eq!(false, g.get_node_value(p1));

        state.restore_state();
        state.save_state();

        g.propagate_node(p3, true, &mut state).unwrap();
        assert!(!g.is_node_bound(d, &state));
        assert!(!g.is_node_bound(p1, &state));
        assert!(!g.is_node_bound(p2, &state));
        assert!(g.is_node_bound(p3, &state));
        assert!(!g.is_node_bound(p4, &state));
        assert_eq!(true, g.get_node_value(p3));

        state.restore_state();
        state.save_state();

        match g.propagate_node(p3, false, &mut state) {
            Ok(_) => (),
            Err(_) => (),
        };
        assert!(g.is_node_bound(d, &state));
        assert!(!g.is_node_bound(d2, &state));
        assert!(!g.is_node_bound(p1, &state));
        assert!(!g.is_node_bound(d3, &state));
        assert!(!g.is_node_bound(p2, &state));
        assert!(g.is_node_bound(p3, &state));
        assert!(!g.is_node_bound(p4, &state));
        assert_eq!(false, g.get_node_value(d));
        assert_eq!(false, g.get_node_value(p1));
        assert_eq!(false, g.get_node_value(p2));
        assert_eq!(false, g.get_node_value(p3));
    }

    #[test]
    fn test_distribution() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        let nodes = g.add_distribution(&vec![0.1, 0.2, 0.7], &mut state);
        let d = g.add_node(false, None, None, &mut state);
        g.add_clause(d, &vec![nodes[0], nodes[2]], &mut state);
        g.add_clause(nodes[1], &vec![d], &mut state);

        match g.propagate_node(nodes[0], true, &mut state) {
            Ok(_) => (),
            Err(_) => (),
        };
        assert!(g.is_node_bound(nodes[0], &state));
        assert!(g.is_node_bound(nodes[1], &state));
        assert!(g.is_node_bound(nodes[2], &state));
        assert_eq!(true, g.get_node_value(nodes[0]));
        assert_eq!(false, g.get_node_value(nodes[1]));
        assert_eq!(false, g.get_node_value(nodes[2]));
    }

    #[test]
    fn test_multiple_implicant_last_false() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        let nodes = (0..3)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(nodes[0], &nodes[1..], &mut state);
        g.propagate_node(NodeIndex(0), false, &mut state).unwrap();
        assert!(g.is_node_bound(NodeIndex(0), &state));
        assert_eq!(false, g.get_node_value(NodeIndex(0)));
        assert!(!g.is_node_bound(NodeIndex(1), &state));
        assert!(!g.is_node_bound(NodeIndex(2), &state));

        g.propagate_node(NodeIndex(1), true, &mut state).unwrap();
        assert!(g.is_node_bound(NodeIndex(0), &state));
        assert_eq!(false, g.get_node_value(NodeIndex(0)));
        assert!(g.is_node_bound(NodeIndex(1), &state));
        assert_eq!(true, g.get_node_value(NodeIndex(1)));
        assert!(g.is_node_bound(NodeIndex(2), &state));
        assert_eq!(false, g.get_node_value(NodeIndex(2)));
    }
}
*/