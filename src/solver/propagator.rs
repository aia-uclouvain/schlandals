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

use crate::core::graph::{ClauseIndex, Graph, NodeIndex};
use crate::core::trail::StateManager;

#[derive(Debug)]
pub struct Unsat;

pub type PropagationResult<T> = Result<T, Unsat>;

/// This is a simple propagator that makes very basic local assumption and propagate them to the
/// graph in a DFS manner.
/// The following rules are enforced
///     1. If a deterministic node has no outgoing or incoming active edges, set it to,
///        respectively, true or false.
///     2. If a node is set to true (resp. false) and is the head (resp. in the body) of the
///        clause, the clause (and all its edges) are deactivated).
///     2. If a node n in the body of a clause (with head h) is set to true, deactivate the edge
///        n->h
///     3. If an active clause has no more active edge, and the head is not bound, set the head to true.
///        In that case all the edges are deactived because their source is true and so the head
///        must be true
///     4. If a node in a distribution is set to true, then all other node in the distribution must
///        be false
///     5. If there is only 1 node unset in a distribution, set it to true
pub trait SimplePropagator {
    /// This is the global propagation algorithm. This run on the whole graph and when the value of
    /// a node can be infered, it launches `propagate_node`.
    fn propagate(&mut self, state: &mut StateManager) -> PropagationResult<f64>;
    /// This implement the propagation of `node` to `value`. It ensures that all the nodes for
    /// which a value can be inferred, after setting `node` to `value` are propagated recursively.
    /// Returns either Unsat if the assignment make the clauses not satisfiable or a float
    /// representing the probability of the assigned node (i.e. the product (or sum in log-space)
    /// of the probability of the probabilistic nodes set to true).
    fn propagate_node(
        &mut self,
        node: NodeIndex,
        value: bool,
        state: &mut StateManager,
    ) -> PropagationResult<f64>;
}

impl SimplePropagator for Graph {
    fn propagate(&mut self, state: &mut StateManager) -> PropagationResult<f64> {
        let mut v = 0.0;
        for node in self.nodes_iter() {
            if !self.is_node_bound(node, state) && self.is_node_deterministic(node) {
                if self.node_number_incoming(node, state) == 0 {
                    v += self.propagate_node(node, false, state)?;
                } else if self.node_number_outgoing(node, state) == 0 {
                    v += self.propagate_node(node, true, state)?;
                }
            }
        }
        PropagationResult::Ok(v)
    }

    fn propagate_node(
        &mut self,
        node: NodeIndex,
        value: bool,
        state: &mut StateManager,
    ) -> PropagationResult<f64> {
        if self.is_node_bound(node, state) {
            if self.get_node_value(node) != value {
                return PropagationResult::Err(Unsat);
            } else {
                return PropagationResult::Ok(0.0);
            }
        }
        self.set_node(node, value, state);
        let mut propagation_prob = if value && self.is_node_probabilistic(node) {
            self.get_node_weight(node).unwrap()
        } else {
            0.0
        };

        let clauses = self
            .node_clauses(node)
            .filter(|clause| self.is_clause_active(*clause, state))
            .collect::<Vec<ClauseIndex>>();
        for clause in clauses {
            let head = self.get_clause_head(clause);
            if (value && node == head) || (!value && node != head) {
                // The clause can be deactivated. That means that each edge is deactivated and
                // the sources/destinations are propagated if necessary
                self.deactivate_clause(clause, state);
                for edge in self.edges_clause(clause) {
                    if self.is_edge_active(edge, state) {
                        self.deactivate_edge(edge, state);
                        let src = self.get_edge_source(edge);
                        if !self.is_node_bound(src, state)
                            && self.is_node_deterministic(src)
                            && self.node_number_outgoing(src, state) == 0
                        {
                            propagation_prob += self.propagate_node(src, true, state)?;
                        }
                        let dst = self.get_edge_destination(edge);
                        if !self.is_node_bound(dst, state)
                            && self.is_node_deterministic(dst)
                            && self.node_number_incoming(dst, state) == 0
                        {
                            propagation_prob += self.propagate_node(dst, false, state)?;
                        }
                    }
                }
            } else {
                if value {
                    let e = self.get_edge_with_implicant(clause, node).unwrap();
                    self.deactivate_edge(e, state);
                }
                // The whole clause can not be deactivated. However, we can still reason about the
                // edges in the clause
                let nb_active_edge = self.clause_number_active_edges(clause, state);
                if value && nb_active_edge == 0 {
                    // The head is not true, but there are no more unassigned implicants. And
                    // they are all true.
                    propagation_prob += self.propagate_node(head, true, state)?;
                } else if nb_active_edge == 1 {
                    let last_active_edge = self.get_first_active_edge(clause, state).unwrap();

                    if self.is_node_bound(head, state) && !self.get_node_value(head) {
                        let n = self.get_edge_source(last_active_edge);
                        propagation_prob += self.propagate_node(n, false, state)?;
                    }
                }
            }
        }
        if self.is_node_probabilistic(node) {
            if value {
                for other in self.nodes_distribution_iter(node).filter(|x| *x != node) {
                    propagation_prob += self.propagate_node(other, false, state)?;
                }
            } else {
                let distribution = self.get_distribution(node).unwrap();
                let number_assigned =
                    self.get_distribution_false_nodes(distribution, state) as usize;
                let number_nodes = self.get_distribution_size(distribution);
                if number_assigned == number_nodes - 1 {
                    // Only 1 node not assigned in the distribution -> set to true
                    for other in self.nodes_distribution_iter(node) {
                        if !self.is_node_bound(other, state) {
                            propagation_prob += self.propagate_node(other, true, state)?;
                            break;
                        }
                    }
                }
            }
        }
        PropagationResult::Ok(propagation_prob)
    }
}

#[cfg(test)]
mod test_simple_propagator_propagation {

    use crate::core::graph::{Graph, NodeIndex};
    use crate::core::trail::StateManager;
    use crate::solver::propagator::SimplePropagator;

    #[test]
    fn initial_propagation_simple_implications() {
        let mut state = StateManager::new();
        let mut g = Graph::new();
        let d: Vec<NodeIndex> = (0..4)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect();
        let p: Vec<NodeIndex> = (0..4)
            .map(|_| g.add_distribution(&vec![0.1], &mut state)[0])
            .collect();

        // deterministic -> deterministic
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
        let mut state = StateManager::new();
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

    #[test]
    fn simple_implications() {
        let mut state = StateManager::new();
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

        g.propagate_node(p1[0], true, &mut state).unwrap();

        assert!(g.is_node_bound(d, &state));
        assert!(g.is_node_bound(p1[0], &state));
        assert!(g.is_node_bound(p2[0], &state));
        assert_eq!(true, g.get_node_value(d));
        assert_eq!(true, g.get_node_value(p1[0]));
        assert_eq!(true, g.get_node_value(p2[0]));

        state.restore_state();
        state.save_state();

        g.propagate_node(p1[0], false, &mut state).unwrap();
        assert!(g.is_node_bound(d, &state));
        assert!(g.is_node_bound(p1[0], &state));
        assert!(!g.is_node_bound(p2[0], &state));
        assert_eq!(false, g.get_node_value(d));
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

        g.propagate_node(p2[0], false, &mut state).unwrap();
        assert!(g.is_node_bound(d, &state));
        assert!(g.is_node_bound(p1[0], &state));
        assert!(g.is_node_bound(p2[0], &state));
        assert_eq!(false, g.get_node_value(d));
        assert_eq!(false, g.get_node_value(p1[0]));
        assert_eq!(false, g.get_node_value(p2[0]));
    }

    #[test]
    fn test_multiple_edges_different_clauses() {
        let mut state = StateManager::new();
        let mut g = Graph::new();
        let d = g.add_node(false, None, None, &mut state);
        let d2 = g.add_node(false, None, None, &mut state);
        let d3 = g.add_node(false, None, None, &mut state);
        let d4 = g.add_node(false, None, None, &mut state);
        let d5 = g.add_node(false, None, None, &mut state);
        let p1 = g.add_distribution(&vec![1.0], &mut state);
        let p2 = g.add_distribution(&vec![1.0], &mut state);
        let p3 = g.add_distribution(&vec![1.0], &mut state);
        let p4 = g.add_distribution(&vec![1.0], &mut state);

        // p1 -        -> p3
        //     |-> d -|
        // p2 -       -> p4

        g.add_clause(p1[0], &vec![d2], &mut state);
        g.add_clause(p2[0], &vec![d3], &mut state);
        g.add_clause(d4, &vec![p3[0]], &mut state);
        g.add_clause(d5, &vec![p4[0]], &mut state);

        g.add_clause(d, &vec![p1[0]], &mut state);
        g.add_clause(d, &vec![p2[0]], &mut state);
        g.add_clause(p3[0], &vec![d], &mut state);
        g.add_clause(p4[0], &vec![d], &mut state);

        state.save_state();

        g.propagate_node(p1[0], true, &mut state).unwrap();
        assert!(g.is_node_bound(d, &state));
        assert!(g.is_node_bound(p1[0], &state));
        assert!(!g.is_node_bound(p2[0], &state));
        assert!(g.is_node_bound(p3[0], &state));
        assert!(g.is_node_bound(p4[0], &state));
        assert_eq!(true, g.get_node_value(d));
        assert_eq!(true, g.get_node_value(p1[0]));
        assert_eq!(true, g.get_node_value(p3[0]));
        assert_eq!(true, g.get_node_value(p4[0]));

        state.restore_state();
        state.save_state();

        g.propagate_node(p1[0], false, &mut state).unwrap();
        assert!(!g.is_node_bound(d, &state));
        assert!(g.is_node_bound(p1[0], &state));
        assert!(!g.is_node_bound(p2[0], &state));
        assert!(!g.is_node_bound(p3[0], &state));
        assert!(!g.is_node_bound(p4[0], &state));
        assert_eq!(false, g.get_node_value(p1[0]));

        state.restore_state();
        state.save_state();

        g.propagate_node(p3[0], true, &mut state).unwrap();
        assert!(!g.is_node_bound(d, &state));
        assert!(!g.is_node_bound(p1[0], &state));
        assert!(!g.is_node_bound(p2[0], &state));
        assert!(g.is_node_bound(p3[0], &state));
        assert!(!g.is_node_bound(p4[0], &state));
        assert_eq!(true, g.get_node_value(p3[0]));

        state.restore_state();
        state.save_state();

        g.propagate_node(p3[0], false, &mut state).unwrap();
        assert!(g.is_node_bound(d, &state));
        assert!(g.is_node_bound(p1[0], &state));
        assert!(g.is_node_bound(p2[0], &state));
        assert!(g.is_node_bound(p3[0], &state));
        assert!(!g.is_node_bound(p4[0], &state));
        assert_eq!(false, g.get_node_value(d));
        assert_eq!(false, g.get_node_value(p1[0]));
        assert_eq!(false, g.get_node_value(p2[0]));
        assert_eq!(false, g.get_node_value(p3[0]));
    }

    #[test]
    fn test_distribution() {
        let mut state = StateManager::new();
        let mut g = Graph::new();
        let nodes = g.add_distribution(&vec![0.1, 0.2, 0.7], &mut state);

        g.propagate_node(nodes[0], true, &mut state).unwrap();
        assert!(g.is_node_bound(nodes[0], &state));
        assert!(g.is_node_bound(nodes[1], &state));
        assert!(g.is_node_bound(nodes[2], &state));
        assert_eq!(true, g.get_node_value(nodes[0]));
        assert_eq!(false, g.get_node_value(nodes[1]));
        assert_eq!(false, g.get_node_value(nodes[2]));
    }

    #[test]
    fn test_multiple_implicant_last_false() {
        let mut state = StateManager::new();
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
