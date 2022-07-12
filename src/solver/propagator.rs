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

/// This is a simple propagator that makes very basic local assumption and propagate them to the
/// graph in a DFS manner.
/// The main ideas of this propagator are the following
///     1. If a deterministic node has no outgoing or incoming active edges, set it to,
///        respectively, true or false.
///     2. If a clause has no more active edge, set the head to true
pub trait SimplePropagator {
    /// This is the global propagation algorithm. This run on the whole graph and when the value of
    /// a node can be infered, it launches `propagate_node`.
    fn propagate(&mut self);
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
    fn propagate_node(&mut self, node: NodeIndex, value: bool);
}

impl SimplePropagator for Graph {
    fn propagate(&mut self) {
        for node in self.nodes_iter() {
            if !self.is_node_bound(node) {
                if self.is_node_deterministic(node) && self.node_number_incoming(node) == 0 {
                    self.propagate_node(node, false);
                } else if self.is_node_deterministic(node) && self.node_number_outgoing(node) == 0 {
                    self.propagate_node(node, true);
                }
            }
        }
    }

    fn propagate_node(&mut self, node: NodeIndex, value: bool) {
        self.set_node(node, value);
        let clauses = self.node_clauses(node).collect::<Vec<ClauseIndex>>();
        for clause in clauses {
            // If
            //     1. The node is assigned to true and is the head of the clause
            //     2. The node is assigned to false and is in the body of the clause
            // then the clause can be deactivated
            let head = self.get_clause_head(clause);

            if (value && head == node) || (!value && head != node) {
                for edge in self.edges_clause(clause) {
                    self.deactivate_edge(edge);
                    let src = self.get_edge_source(edge);
                    let dst = self.get_edge_destination(edge);
                    if self.is_node_deterministic(src)
                        && src != node
                        && self.node_number_outgoing(src) == 0
                        && !self.is_node_bound(src)
                    {
                        self.propagate_node(src, true);
                    }
                    if self.is_node_deterministic(dst)
                        && dst != node
                        && self.node_number_incoming(dst) == 0
                        && !self.is_node_bound(dst)
                    {
                        self.propagate_node(dst, false);
                    }
                }
            } else if value {
                // The node is assigned to true, but is not the head of the clause. In this case,
                // the edge node -> h is deactivated
                for edge in self.edges_clause(clause) {
                    let src = self.get_edge_source(edge);
                    if src == node {
                        self.deactivate_edge(edge);
                        if self.clause_number_active_edges(clause) == 0 && !self.is_node_bound(head)
                        {
                            // All the literals in the implicants are set to true, so the head must
                            // be true
                            self.propagate_node(head, true);
                        }
                        break;
                    }
                }
            } else if !value {
                // The node is assigned to false, but is the head of the clause. Thus the body
                // needs to evaluate to false. There is nothing to do expect if there are only one
                // literal left unassigned in the implicant
                if self.clause_number_active_edges(clause) == 1 {
                    for edge in self.edges_clause(clause) {
                        if self.is_edge_active(edge) {
                            let src = self.get_edge_source(edge);
                            // This must be the last active edge
                            if !self.is_node_bound(src) {
                                self.propagate_node(src, false);
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test_simple_propagator_propagation {

    use crate::core::graph::{Graph, NodeIndex};
    use crate::solver::propagator::SimplePropagator;

    #[test]
    fn initial_propagation_simple_implications() {
        let mut g = Graph::new();
        let d: Vec<NodeIndex> = (0..4).map(|_| g.add_node(false, None)).collect();
        let p: Vec<NodeIndex> = (0..4).map(|_| g.add_node(true, Some(0.1))).collect();

        // deterministic -> deterministic
        // d[1] -> d[0]
        g.add_clause(d[0], &vec![d[1]]);
        // deterministic -> probabilistic
        // d[2] -> p[0]
        g.add_clause(p[0], &vec![d[2]]);
        // probabilistic -> deterministic
        // p[1] -> d[3]
        g.add_clause(d[3], &vec![p[1]]);
        // probabilistic -> probabilistic
        // p[2] -> p[3]
        g.add_clause(p[3], &vec![p[2]]);

        g.propagate();
        assert!(g.is_node_bound(d[0]));
        assert!(g.is_node_bound(d[1]));
        assert!(g.is_node_bound(d[2]));
        assert!(g.is_node_bound(d[3]));
        assert_eq!(true, g.get_node_value(d[0]));
        assert_eq!(true, g.get_node_value(d[1]));
        assert_eq!(false, g.get_node_value(d[2]));
        assert_eq!(true, g.get_node_value(d[3]));
    }

    #[test]
    fn initial_propagation_chained_implications() {
        let mut g = Graph::new();
        let d: Vec<NodeIndex> = (0..6).map(|_| g.add_node(false, None)).collect();
        let p: Vec<NodeIndex> = (0..3).map(|_| g.add_node(true, Some(0.1))).collect();

        // d[0] -> d[1] -> d[2]
        g.add_clause(d[1], &vec![d[0]]);
        g.add_clause(d[2], &vec![d[1]]);

        // d[3] -> p[0] -> d[4]
        g.add_clause(p[0], &vec![d[3]]);
        g.add_clause(d[4], &vec![p[0]]);

        // p[1] -> d[5] -> p[2]
        g.add_clause(d[5], &vec![p[1]]);
        g.add_clause(p[2], &vec![d[5]]);

        g.propagate();

        assert!(g.is_node_bound(d[0]));
        assert!(g.is_node_bound(d[1]));
        assert!(g.is_node_bound(d[2]));
        assert!(g.is_node_bound(d[3]));
        assert!(g.is_node_bound(d[4]));
        assert!(!g.is_node_bound(d[5]));
        assert!(!g.is_node_bound(p[0]));
        assert!(!g.is_node_bound(p[1]));
        assert!(!g.is_node_bound(p[2]));

        assert_eq!(false, g.get_node_value(d[0]));
        assert_eq!(false, g.get_node_value(d[1]));
        assert_eq!(false, g.get_node_value(d[2]));
        assert_eq!(false, g.get_node_value(d[3]));
        assert_eq!(true, g.get_node_value(d[4]));
    }
}

#[cfg(test)]
mod test_simple_propagataor_node_propagation {
    use crate::core::graph::Graph;
    use crate::core::trail::SaveAndRestore;
    use crate::solver::propagator::SimplePropagator;

    #[test]
    fn simple_implications() {
        let mut g = Graph::new();
        let d = g.add_node(false, None);
        let p1 = g.add_node(true, Some(0.4));
        let p2 = g.add_node(true, Some(0.1));

        // p1 -> d -> p2
        g.add_clause(d, &vec![p1]);
        g.add_clause(p2, &vec![d]);

        g.state.save_state();

        g.propagate_node(p1, true);

        assert!(g.is_node_bound(d));
        assert!(g.is_node_bound(p1));
        assert!(g.is_node_bound(p2));
        assert_eq!(true, g.get_node_value(d));
        assert_eq!(true, g.get_node_value(p1));
        assert_eq!(true, g.get_node_value(p2));

        g.state.restore_state();
        g.state.save_state();

        g.propagate_node(p1, false);
        assert!(g.is_node_bound(d));
        assert!(g.is_node_bound(p1));
        assert!(!g.is_node_bound(p2));
        assert_eq!(false, g.get_node_value(d));
        assert_eq!(false, g.get_node_value(p1));

        g.state.restore_state();
        g.state.save_state();

        g.propagate_node(p2, true);
        assert!(g.is_node_bound(d));
        assert!(!g.is_node_bound(p1));
        assert!(g.is_node_bound(p2));
        assert_eq!(true, g.get_node_value(d));
        assert_eq!(true, g.get_node_value(p2));

        g.state.restore_state();
        g.state.save_state();

        g.propagate_node(p2, false);
        assert!(g.is_node_bound(d));
        assert!(g.is_node_bound(p1));
        assert!(g.is_node_bound(p2));
        assert_eq!(false, g.get_node_value(d));
        assert_eq!(false, g.get_node_value(p1));
        assert_eq!(false, g.get_node_value(p2));
    }

    #[test]
    fn test_multiple_edges_different_clauses() {
        let mut g = Graph::new();
        let d = g.add_node(false, None);
        let p1 = g.add_node(true, Some(0.4));
        let p2 = g.add_node(true, Some(0.1));
        let p3 = g.add_node(true, Some(0.3));
        let p4 = g.add_node(true, Some(0.25));

        // p1 -        -> p3
        //     |-> d -|
        // p2 -       -> p4

        g.add_clause(d, &vec![p1]);
        g.add_clause(d, &vec![p2]);
        g.add_clause(p3, &vec![d]);
        g.add_clause(p4, &vec![d]);

        g.state.save_state();

        g.propagate_node(p1, true);
        assert!(g.is_node_bound(d));
        assert!(g.is_node_bound(p1));
        assert!(!g.is_node_bound(p2));
        assert!(g.is_node_bound(p3));
        assert!(g.is_node_bound(p4));
        assert_eq!(true, g.get_node_value(d));
        assert_eq!(true, g.get_node_value(p1));
        assert_eq!(true, g.get_node_value(p3));
        assert_eq!(true, g.get_node_value(p4));

        g.state.restore_state();
        g.state.save_state();

        g.propagate_node(p1, false);
        assert!(!g.is_node_bound(d));
        assert!(g.is_node_bound(p1));
        assert!(!g.is_node_bound(p2));
        assert!(!g.is_node_bound(p3));
        assert!(!g.is_node_bound(p4));
        assert_eq!(false, g.get_node_value(p1));

        g.state.restore_state();
        g.state.save_state();

        g.propagate_node(p3, true);
        assert!(!g.is_node_bound(d));
        assert!(!g.is_node_bound(p1));
        assert!(!g.is_node_bound(p2));
        assert!(g.is_node_bound(p3));
        assert!(!g.is_node_bound(p4));
        assert_eq!(true, g.get_node_value(p3));

        g.state.restore_state();
        g.state.save_state();

        g.propagate_node(p3, false);
        assert!(g.is_node_bound(d));
        assert!(g.is_node_bound(p1));
        assert!(g.is_node_bound(p2));
        assert!(g.is_node_bound(p3));
        assert!(!g.is_node_bound(p4));
        assert_eq!(false, g.get_node_value(d));
        assert_eq!(false, g.get_node_value(p1));
        assert_eq!(false, g.get_node_value(p2));
        assert_eq!(false, g.get_node_value(p3));
    }
}
