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
///     2. If a node n in the body of a clause (with head h) is set to true, deactivate the edge
///        n->h
///     3. If a clause has no more active edge, and the head is not bound, set the head to true
///     4. If a node in a distribution is set to true, then all other node in the distribution must
///        be false
///     5. If there is only 1 node unset in a distribution, set it to true
pub trait SimplePropagator {
    /// This is the global propagation algorithm. This run on the whole graph and when the value of
    /// a node can be infered, it launches `propagate_node`.
    fn propagate<S: StateManager>(&mut self, state: &mut S) -> PropagationResult<f64>;
    /// Implements de propagator logics when `node` is set to `value`. This breaks down as follows
    ///
    /// case 1: `value = true`:
    ///     - All the clauses B => h with h = `node` are deactivated. This is fine because the
    ///     implication is always respected, whatever choice is made for the node in B.
    ///     - For all clauses B => h such that `node` is in B, the edge `node` -> h is deactivated.
    ///     This is equivalent to "removing" `node` from the body of the clause. If the number of
    ///     active edges in the clause becomes 0, then we have that h **must** be true since in
    ///     that case all literal in B are true.
    ///
    /// case 2: `value = false`:
    ///     - All the clauses B => h with `node` in B can be deactivated, for the same reasons as
    ///     in case 1
    ///     - For all clause B => `node`, at least one literal in the body must be set to false. If
    ///     the body has only one literal, it is set to false.
    ///
    /// While deactivating the edges, the following conditions are checked:
    ///     1. If a deterministic node has no more outgoing edges, it is set to true. In this case,
    ///        the node appears only as head in its clauses. Setting it to true will never yield
    ///        interpretation that are not models and thus can be done safely
    ///     2. If a deterministic node has no more incoming edges, it is set to false. In this
    ///        case, the node only appears in the body of the clauses, and by the same reasoning
    ///        setting it to false is safe.
    ///     3. If a clause B => h with h set to false has only one active edge remaining, the
    ///        source of this edge is set to false. In that case the clause is still active, and
    ///        all other literals in the body are set to true. In order to be a model, the last one
    ///        must be false
    ///         
    fn propagate_node<S: StateManager>(
        &mut self,
        node: NodeIndex,
        value: bool,
        state: &mut S,
    ) -> PropagationResult<f64>;
}

impl SimplePropagator for Graph {
    fn propagate<S: StateManager>(&mut self, state: &mut S) -> PropagationResult<f64> {
        let mut v = 0.0;
        for node in self.nodes_iter() {
            if !self.is_node_bound(node, state) {
                if self.is_node_deterministic(node) && self.node_number_incoming(node, state) == 0 {
                    v += self.propagate_node(node, false, state)?;
                } else if self.is_node_deterministic(node)
                    && self.node_number_outgoing(node, state) == 0
                {
                    v += self.propagate_node(node, true, state)?;
                }
            }
        }
        PropagationResult::Ok(v)
    }

    fn propagate_node<S: StateManager>(
        &mut self,
        node: NodeIndex,
        value: bool,
        state: &mut S,
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

        let clauses = self.node_clauses(node).collect::<Vec<ClauseIndex>>();
        for clause in clauses {
            let head = self.get_clause_head(clause);
            if (value && node == head) || (!value && node != head) {
                // The clause can be deactivated. That means that each edge is deactivated and
                // the sources/destinations are propagated if necessary
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
                if value && nb_active_edge == 0 && !self.is_node_bound(head, state) {
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
            let distribution = self.get_distribution(node).unwrap();
            if value {
                for other in self.nodes_distribution_iter(node).filter(|x| *x != node) {
                    propagation_prob += self.propagate_node(other, false, state)?;
                }
            } else if self.get_distribution_false_nodes(distribution, state) as usize
                == self.get_distribution_size(distribution) - 1
            {
                for other in self.nodes_distribution_iter(node) {
                    if !self.is_node_bound(other, state) {
                        propagation_prob += self.propagate_node(other, true, state)?;
                        break;
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
    use crate::core::trail::TrailedStateManager;
    use crate::solver::propagator::SimplePropagator;

    #[test]
    fn initial_propagation_simple_implications() {
        let mut state = TrailedStateManager::new();
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
        let mut state = TrailedStateManager::new();
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
    use crate::core::trail::{SaveAndRestore, TrailedStateManager};
    use crate::solver::propagator::SimplePropagator;

    #[test]
    fn simple_implications() {
        let mut state = TrailedStateManager::new();
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
        let mut state = TrailedStateManager::new();
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
        let mut state = TrailedStateManager::new();
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
        let mut state = TrailedStateManager::new();
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
