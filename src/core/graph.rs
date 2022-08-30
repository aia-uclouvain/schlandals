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

//! This module implements a graph using vector indexes. This implementation is inspired by [this
//! blog post](https://smallcultfollowing.com/babysteps/blog/2015/04/06/modeling-graphs-in-rust-using-vector-indices/)
//! and [this
//! code](https://github.com/xgillard/ddo/blob/master/src/implementation/mdd/deep/mddgraph.rs).
//!
//! A graph G = (V, E) is represented using two vectors containing, respectively, all the nodes and
//! all the edges. A node (or an edge) is identified uniquely by its index in its enclosing vector.
//! The `NodeIndex` and `EdgeIndex` structures are used to index the nodes and edges.
//!
//! The parents and children of each node is implemented as a succession of 'pointers' of
//! `EdgeIndex`.
//! If a node n1 has two children n2, n3 then there are two directed edges in the graph n1 -> n2
//! and n1 -> n3.
//! These edges are respectively indexed by e1 and e2.
//! In the `NodeData` structure, the field `children` is filled with the value `Some(e1)`, which
//! references the first of its outgoing edges (to its children n2).
//! In the `EdgeData` for the edge e1, the field `next_outgoing` is set to `Some(e2)`, the second
//! outgoing edge of n1.
//! On the other hand, since there are no more child to n1 after n3, this field is `None` for the
//! edge identified by e2.
//!
//! # Note:
//! Once the graph is constructed, no edge/node should be removed from it. Thus this
//! implementation does not have problems like dangling indexes.

use super::trail::*;

// The following abstractions allow to have type safe indexing for the nodes, edes and clauses.
// They are used to retrieve respectively `NodeData`, `EdgeData` and `Clause` in the `Graph`
// structure.

/// Abstraction used as a typesafe way of retrieving a `NodeData` in the `Graph` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NodeIndex(pub usize);

/// Abstraction used as a typesafe way of retrieving a `EdgeData` in the `Graph` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct EdgeIndex(pub usize);

/// Abstraction used as a typesafe way of retrieving a `Clause` in the `Graph` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ClauseIndex(pub usize);

/// Abstraction used as a typesafe way of retrieving a `Distribution` in the `Graph` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DistributionIndex(pub usize);

/// This structure represent a clause in the graph. A clause is of the form
///     a && b && ... && d => e
/// In the graph, this will be represented by n (the number of literals in the implicant) incoming
/// edges in `e`.
/// The edges of a clause are added at the same time, so a clause can be fully identified by an
/// `EdgeIndex` (the first edge, in the example `a -> e`) and the size of the clause (n)
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Clause {
    /// Index of the first edge of the clause in the `edges` vector in `Graph`
    pub first: EdgeIndex,
    /// Number of edges in the clause (i.e. size of the body)
    pub size: usize,
    /// Head of the clause
    pub head: NodeIndex,
    /// Number of active edges in the clause
    active_edges: ReversibleInt,
}

/// Data structure that actually holds the data of a  node in the graph
/// A node in the graph is in four possible states: 1) Unassigned 2) True 3) False 4)
/// Unconstrained
///
/// In the last case, it means that the node can be either `true` or `false` without impacting the
/// counting.
/// The value of the node is stored in the `value` field and its domain is implicitly given by the
/// `domain_size` field.
/// If a `domain_size = 2` then both `true` and `false` are in the domain, and the variable is
/// unassgined. If `domain_size = 1` then the value is assigned to the value in the `value` field.
///
/// # Note:
/// This might not be the best design, but it seems that a full handling of domain etc (like in
/// a cp solver) is a bit overkill since at the moment we only need BoolVar.
#[derive(Debug, Copy, Clone)]
pub struct NodeData {
    /// The value assigned to the node. Should only be read when `domain_size = 1`
    pub value: bool,
    /// The current size of the domain of the node
    domain_size: ReversibleInt,

    /// Indicate if the literal represented by the node is a probabilistic literal (i.e. have a
    /// weight) or not
    pub probabilistic: bool,

    /// If `probabilistic` is `true`, then this is the weight associated to the node. Otherwise
    /// this is None. The weight is assumed to be log-transformed.
    pub weight: Option<f64>,

    /// If present, the index of the first outgoing edge of the node
    pub children: Option<EdgeIndex>,
    /// If present, the index of the first incoming edge of the node
    pub parents: Option<EdgeIndex>,
    /// Number of active incoming edges
    active_incoming: ReversibleInt,
    /// Numbre of active outgoing edges
    active_outgoing: ReversibleInt,

    /// If `probabilistic` is `true`, this is the index of the distribution containing this node
    pub distribution: Option<DistributionIndex>,
}

/// Data structure that actually holds the data of an edge in the graph
#[derive(Debug, Copy, Clone)]
pub struct EdgeData {
    /// The index of the source node in the Graph
    pub src: NodeIndex,

    /// The index of the target node in the Graph
    pub dst: NodeIndex,

    /// If exists, the next incoming edge of `dst`
    pub next_incoming: Option<EdgeIndex>,

    /// If exists, the next outgoing edge of `src`
    pub next_outgoing: Option<EdgeIndex>,

    // The clause in which this edges is
    clause: ClauseIndex,
    /// Is the edge active or not
    active: ReversibleBool,
}

/// Represents a set of nodes in a same distribution. This assume that the nodes of a distribution
/// are inserted in the graph one after the other (i.e. that their `NodeIndex` are consecutive).
/// Since no node should be removed from the graph once constructed, this should not be a problem.
/// Thus a distribution is identified by the first `NodeIndex` and the number of nodes in the
/// distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Distribution {
    /// First node in the distribution
    pub first: NodeIndex,
    /// Number of node in the distribution
    pub size: usize,
    /// Number of active edges in the distribution. This is the sum of the active ingoing/outgoing
    /// edges of all nodes of the distribution
    active_edges: ReversibleInt,
    /// Number of nodes set to false in the distribution
    nodes_false: ReversibleInt,
}

/// Data structure representing the Graph.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Vector containing the nodes of the graph
    nodes: Vec<NodeData>,
    /// Vector containing the edges of the graph
    edges: Vec<EdgeData>,
    /// Vector containing the clauses of the graph
    clauses: Vec<Clause>,
    /// Objective of the graph. This is the sum of all probabilistic nodes set to true
    pub obj: ReversibleFloat,
    /// Vector containing the distributions of the graph
    distributions: Vec<Distribution>,
}

impl Graph {
    pub fn new<S: StateManager>(state: &mut S) -> Self {
        let obj = state.manage_float(0.0);
        Self {
            nodes: vec![],
            edges: vec![],
            clauses: vec![],
            obj,
            distributions: vec![],
        }
    }

    /// Returns the current objective with the assignment of the nodes in the graph
    pub fn get_objective<S: StateManager>(&self, state: &S) -> f64 {
        state.get_float(self.obj)
    }

    /// Returns the number of node in the graph
    pub fn number_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of distribution in the graph
    pub fn number_distributions(&self) -> usize {
        self.distributions.len()
    }

    /// Returns the distribution of a node if ane
    pub fn get_distribution(&self, node: NodeIndex) -> Option<DistributionIndex> {
        self.nodes[node.0].distribution
    }

    // --- Methods that returns an iterator --- ///

    /// Returns an iterator on the node indexes (`NodeIndex`)
    pub fn nodes_iter(&self) -> Nodes {
        Nodes {
            limit: self.nodes.len(),
            next: 0,
        }
    }

    /// Returns an iterator over all the nodes in the distribution identified by `distribution`
    pub fn distribution_iter(
        &self,
        distribution: DistributionIndex,
    ) -> impl Iterator<Item = NodeIndex> {
        let start = self.distributions[distribution.0].first.0;
        let size = self.distributions[distribution.0].size;
        let end = start + size;
        (start..end).map(|i| NodeIndex(i))
    }

    /// Returns an iterator on the node in the same distribution as `node`
    pub fn nodes_distribution_iter(&self, node: NodeIndex) -> DistributionIterator {
        if self.is_node_deterministic(node) {
            DistributionIterator {
                limit: node.0 + 1,
                next: node.0,
            }
        } else {
            let distribution = self.nodes[node.0].distribution.unwrap();
            let next = self.distributions[distribution.0].first.0;
            let limit = next + self.distributions[distribution.0].size;
            DistributionIterator { limit, next }
        }
    }

    /// Returns an iterator over the outgoing edges of `node`
    pub fn outgoings(&self, node: NodeIndex) -> Outgoings<'_> {
        let first = self.nodes[node.0].children;
        Outgoings {
            graph: self,
            next: first,
        }
    }

    /// Returns an iterator over the incoming edges of `node`
    pub fn incomings(&self, node: NodeIndex) -> Incomings<'_> {
        let first = self.nodes[node.0].parents;
        Incomings {
            graph: self,
            next: first,
        }
    }

    /// Returns an iterator over all the clauses in which `node` is included, either as the head of
    /// the clauses or in its body
    pub fn node_clauses(&self, node: NodeIndex) -> impl Iterator<Item = ClauseIndex> + Sized + '_ {
        // A bit ugly... For now dedup is not available for iterators (see
        // https://github.com/rust-lang/rust/pull/83748)
        // The idea is that edges are inserted by clauses, and thus all incoming edges of the same clauses
        // will be neighbors in the linked list of incoming edges. Thus mapping the edges to their
        // clause will yield a sequence of `ClauseIndex` sorted in decreasing order
        let mut incomings: Vec<ClauseIndex> = self
            .incomings(node)
            .map(move |e| self.edges[e.0].clause)
            .collect();
        incomings.dedup();
        let outgoings = self.outgoings(node).map(move |e| self.edges[e.0].clause);
        incomings.into_iter().chain(outgoings)
    }

    /// Returns an iterator over all the `EdgeIndex` of a clause
    pub fn edges_clause(&self, clause: ClauseIndex) -> EdgesClause {
        let first = self.clauses[clause.0].first.0;
        let limit = first + self.clauses[clause.0].size;
        EdgesClause { limit, next: first }
    }

    // --- End iterator methods --- ///

    // --- Node related methods --- ///

    /// Add a node to the graph. At the creation of the graph, the nodes do not have any value. If
    /// the node represent a probabilistic literal (`probabilistic = true`), then it has a weight
    /// of `weight`. This method returns the index of the node.
    pub fn add_node<S: StateManager>(
        &mut self,
        probabilistic: bool,
        weight: Option<f64>,
        distribution: Option<DistributionIndex>,
        state: &mut S,
    ) -> NodeIndex {
        let id = self.nodes.len();
        self.nodes.push(NodeData {
            value: false,
            domain_size: state.manage_int(2),
            probabilistic,
            weight,
            children: None,
            parents: None,
            active_incoming: state.manage_int(0),
            active_outgoing: state.manage_int(0),
            distribution,
        });
        NodeIndex(id)
    }

    /// Add a distribution to the graph. In this case, a distribution is a set of probabilistic
    /// nodes such that
    ///     - The sum of their weights sum up to 1.0
    ///     - Only one of these node can be true at a given time
    ///     - None of the node in the distribution is part of another distribution
    ///
    /// Each probabilstic node should be part of one distribution.
    pub fn add_distribution<S: StateManager>(
        &mut self,
        weights: &Vec<f64>,
        state: &mut S,
    ) -> Vec<NodeIndex> {
        let distribution = DistributionIndex(self.distributions.len());
        let nodes: Vec<NodeIndex> = weights
            .iter()
            .map(|w| self.add_node(true, Some(*w), Some(distribution), state))
            .collect();
        self.distributions.push(Distribution {
            first: nodes[0],
            size: nodes.len(),
            active_edges: state.manage_int(0),
            nodes_false: state.manage_int(0),
        });
        nodes
    }

    /// Gets the number of active edges of a distribution
    pub fn get_distribution_number_active_edges<S: StateManager>(
        &self,
        distribution: DistributionIndex,
        state: &S,
    ) -> isize {
        state.get_int(self.distributions[distribution.0].active_edges)
    }

    /// Gets the number of nodes set to false in a distribution
    pub fn get_distribution_false_nodes<S: StateManager>(
        &self,
        distribution: DistributionIndex,
        state: &S,
    ) -> isize {
        state.get_int(self.distributions[distribution.0].nodes_false)
    }

    /// Gets the number of nodes in a distribution
    pub fn get_distribution_size(&self, distribution: DistributionIndex) -> usize {
        self.distributions[distribution.0].size
    }

    /// Sets `node` to `value`. This assumes that `node` is unassigned
    /// `node` is probabilistic and `value` is true, then the probability of the node is added to
    /// the current objective
    pub fn set_node<S: StateManager>(&mut self, node: NodeIndex, value: bool, state: &mut S) {
        // TODO: Maybe would be useful to launch an error
        debug_assert!(state.get_int(self.nodes[node.0].domain_size) == 2);
        // If the node is probabilistic, add its value to the objective
        let n = &self.nodes[node.0];
        if n.probabilistic {
            // If the node is set to true, then its weight is added to the objective. If not, then
            // the counter of node set to false for the distribution of `node` is incremented
            if value {
                state.add_float(self.obj, n.weight.unwrap());
            } else {
                let distribution = self.get_distribution(node).unwrap();
                state.increment(self.distributions[distribution.0].nodes_false);
            }
        }
        // Assigning the value to the node
        state.decrement(self.nodes[node.0].domain_size);
        self.nodes[node.0].value = value;
    }

    /// Returns true if `node` is bound to a value, false otherwise
    pub fn is_node_bound<S: StateManager>(&self, node: NodeIndex, state: &S) -> bool {
        state.get_int(self.nodes[node.0].domain_size) == 1
    }

    /// Gets the value assigned to a node
    pub fn get_node_value(&self, node: NodeIndex) -> bool {
        self.nodes[node.0].value
    }

    /// Returns true if the node is deterministic and false otherwise
    pub fn is_node_deterministic(&self, node: NodeIndex) -> bool {
        !self.nodes[node.0].probabilistic
    }

    /// Returns true if the node is probabilistic and false otherwise
    pub fn is_node_probabilistic(&self, node: NodeIndex) -> bool {
        !self.is_node_deterministic(node)
    }

    /// Returns the number of active incoming edges of `node`
    pub fn node_number_incoming<S: StateManager>(&self, node: NodeIndex, state: &S) -> isize {
        state.get_int(self.nodes[node.0].active_incoming)
    }

    /// Returns the number of active outgoing edges of `node`
    pub fn node_number_outgoing<S: StateManager>(&self, node: NodeIndex, state: &S) -> isize {
        state.get_int(self.nodes[node.0].active_outgoing)
    }

    // --- End node related methods --- //

    // --- Edge related methods --- //

    /// Add an edge between the node identified by `src` to the node identified by `dst`. This
    /// method returns the index of the edge.
    fn add_edge<S: StateManager>(
        &mut self,
        src: NodeIndex,
        dst: NodeIndex,
        clause: ClauseIndex,
        state: &mut S,
    ) -> EdgeIndex {
        let source_outgoing = self.nodes[src.0].children;
        let dest_incoming = self.nodes[dst.0].parents;
        let edge = EdgeData {
            src,
            dst,
            next_incoming: dest_incoming,
            next_outgoing: source_outgoing,
            clause,
            active: state.manage_boolean(true),
        };
        let index = EdgeIndex(self.edges.len());
        self.nodes[src.0].children = Some(index);
        state.increment(self.nodes[src.0].active_outgoing);
        self.nodes[dst.0].parents = Some(index);
        state.increment(self.nodes[dst.0].active_incoming);
        self.edges.push(edge);
        if self.is_node_probabilistic(src) {
            let src_distribution = self.get_distribution(src).unwrap();
            state.increment(self.distributions[src_distribution.0].active_edges);
        }
        if self.is_node_probabilistic(dst) {
            let dst_distribution = self.get_distribution(dst).unwrap();
            state.increment(self.distributions[dst_distribution.0].active_edges);
        }
        index
    }

    /// Deactivate `edge` and decrements the numbre of active incoming edges of `edge.dst` as well as
    /// the number of outgoing edges of `edge.src`
    pub fn deactivate_edge<S: StateManager>(&mut self, edge: EdgeIndex, state: &mut S) {
        let edge = self.edges[edge.0];
        state.set_bool(edge.active, false);
        state.decrement(self.nodes[edge.src.0].active_outgoing);
        state.decrement(self.nodes[edge.dst.0].active_incoming);
        state.decrement(self.clauses[edge.clause.0].active_edges);
        // If the source or the destination are probabilistic, decrement the counter of active
        // edges in their distribution
        if self.is_node_probabilistic(edge.src) {
            let distribution = self.get_distribution(edge.src).unwrap();
            state.decrement(self.distributions[distribution.0].active_edges);
        }
        if self.is_node_probabilistic(edge.dst) {
            let distribution = self.get_distribution(edge.dst).unwrap();
            state.decrement(self.distributions[distribution.0].active_edges);
        }
    }

    /// Return true if the edge is still active
    pub fn is_edge_active<S: StateManager>(&self, edge: EdgeIndex, state: &S) -> bool {
        state.get_bool(self.edges[edge.0].active)
    }

    /// Returns the source of the edge
    pub fn get_edge_source(&self, edge: EdgeIndex) -> NodeIndex {
        self.edges[edge.0].src
    }

    /// Returns the destination of the edge
    pub fn get_edge_destination(&self, edge: EdgeIndex) -> NodeIndex {
        self.edges[edge.0].dst
    }

    // --- End edge related methods --- //

    // --- Clause related methods --- //

    /// Add a clause to the graph. A clause is a expression of the form
    ///     n1 && n2 && ... && nn => head
    ///
    /// where head, n1, ..., nn are nodes of the graph. ´head` is the head of the clause and `body`
    /// = vec![n1, ..., nn].
    /// This function adds n edges to the graph, one for each node in the body (as source) towards
    /// the head of the clause
    pub fn add_clause<S: StateManager>(
        &mut self,
        head: NodeIndex,
        body: &[NodeIndex],
        state: &mut S,
    ) -> ClauseIndex {
        let cid = ClauseIndex(self.clauses.len());
        self.clauses.push(Clause {
            first: EdgeIndex(self.edges.len()),
            size: body.len(),
            head,
            active_edges: state.manage_int(body.len() as isize),
        });
        for node in body {
            self.add_edge(*node, head, cid, state);
        }
        cid
    }

    /// Returns the number of active edges in the clause
    pub fn clause_number_active_edges<S: StateManager>(
        &self,
        clause: ClauseIndex,
        state: &S,
    ) -> isize {
        state.get_int(self.clauses[clause.0].active_edges)
    }

    /// Returns the head of the clause
    pub fn get_clause_head(&self, clause: ClauseIndex) -> NodeIndex {
        self.clauses[clause.0].head
    }
}

// --- ITERATORS --- //

pub struct Outgoings<'g> {
    graph: &'g Graph,
    next: Option<EdgeIndex>,
}

impl<'g> Iterator for Outgoings<'g> {
    type Item = EdgeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next {
            None => None,
            Some(eidx) => {
                let edge = &self.graph.edges[eidx.0];
                self.next = edge.next_outgoing;
                Some(eidx)
            }
        }
    }
}

pub struct Incomings<'g> {
    graph: &'g Graph,
    next: Option<EdgeIndex>,
}

impl<'g> Iterator for Incomings<'g> {
    type Item = EdgeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next {
            None => None,
            Some(eidx) => {
                let edge = &self.graph.edges[eidx.0];
                self.next = edge.next_incoming;
                Some(eidx)
            }
        }
    }
}

pub struct DistributionIterator {
    limit: usize,
    next: usize,
}

impl Iterator for DistributionIterator {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.limit {
            None
        } else {
            self.next += 1;
            Some(NodeIndex(self.next - 1))
        }
    }
}

impl IntoIterator for Distribution {
    type Item = NodeIndex;
    type IntoIter = DistributionIterator;

    fn into_iter(self) -> Self::IntoIter {
        DistributionIterator {
            limit: self.first.0 + self.size,
            next: self.first.0,
        }
    }
}

pub struct Nodes {
    limit: usize,
    next: usize,
}

impl Iterator for Nodes {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.limit {
            None
        } else {
            self.next += 1;
            Some(NodeIndex(self.next - 1))
        }
    }
}

pub struct EdgesClause {
    limit: usize,
    next: usize,
}

impl Iterator for EdgesClause {
    type Item = EdgeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.limit {
            None
        } else {
            self.next += 1;
            Some(EdgeIndex(self.next - 1))
        }
    }
}

// --- END ITERATORS --- //

#[cfg(test)]
mod test_node_data {
    use crate::core::graph::{DistributionIndex, NodeData};
    use crate::core::trail::*;

    #[test]
    fn new_can_create_probabilistic_node() {
        let mut state = TrailedStateManager::new();
        let node = NodeData {
            value: true,
            domain_size: state.manage_int(2),
            probabilistic: true,
            weight: Some(0.4),
            children: None,
            parents: None,
            active_outgoing: state.manage_int(0),
            active_incoming: state.manage_int(0),
            distribution: Some(DistributionIndex(0)),
        };
        assert_eq!(true, node.probabilistic);
        assert!(node.weight.is_some());
        assert_eq!(0.4, node.weight.unwrap());
    }

    #[test]
    fn new_can_create_deterministic_node() {
        let mut state = TrailedStateManager::new();
        let node = NodeData {
            value: true,
            domain_size: state.manage_int(2),
            probabilistic: false,
            weight: None,
            children: None,
            parents: None,
            active_outgoing: state.manage_int(0),
            active_incoming: state.manage_int(0),
            distribution: None,
        };
        assert_eq!(false, node.probabilistic);
        assert_eq!(None, node.weight);
    }
}

#[cfg(test)]
mod test_edge_data {
    use crate::core::graph::{ClauseIndex, EdgeData, EdgeIndex, NodeIndex};
    use crate::core::trail::*;

    #[test]
    fn new_can_create_first() {
        let src = NodeIndex(11);
        let dst = NodeIndex(13);
        let mut state = TrailedStateManager::new();
        let edge = EdgeData {
            src,
            dst,
            next_incoming: None,
            next_outgoing: None,
            clause: ClauseIndex(0),
            active: state.manage_boolean(true),
        };
        assert_eq!(src, edge.src);
        assert_eq!(dst, edge.dst);
        assert_eq!(None, edge.next_incoming);
        assert_eq!(None, edge.next_outgoing);
    }

    #[test]
    fn new_can_create_not_first() {
        let src = NodeIndex(42);
        let dst = NodeIndex(64);
        let next_incoming = Some(EdgeIndex(3));
        let next_outgoing = Some(EdgeIndex(102));
        let mut state = TrailedStateManager::new();
        let edge = EdgeData {
            src,
            dst,
            next_incoming: Some(EdgeIndex(3)),
            next_outgoing: Some(EdgeIndex(102)),
            clause: ClauseIndex(0),
            active: state.manage_boolean(true),
        };
        assert_eq!(next_incoming, edge.next_incoming);
        assert_eq!(next_outgoing, edge.next_outgoing);
    }
}

#[cfg(test)]
mod test_graph {
    use crate::core::graph::*;

    #[test]
    fn new_create_empty_graph() {
        let mut state = TrailedStateManager::new();
        let g = Graph::new(&mut state);
        assert_eq!(0, g.nodes.len());
        assert_eq!(0, g.edges.len());
    }

    #[test]
    fn add_nodes_increment_index() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        for i in 0..10 {
            let idx = g.add_node(false, None, None, &mut state);
            assert_eq!(NodeIndex(i), idx);
        }
    }

    #[test]
    fn add_edges_increment_index() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        g.add_node(false, None, None, &mut state);
        g.add_node(false, None, None, &mut state);
        for i in 0..10 {
            let idx = g.add_edge(NodeIndex(0), NodeIndex(1), ClauseIndex(0), &mut state);
            assert_eq!(EdgeIndex(i), idx);
        }
    }

    #[test]
    fn add_multiple_incoming_edges_to_node() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        let n1 = g.add_node(false, None, None, &mut state);
        let n2 = g.add_node(false, None, None, &mut state);
        let n3 = g.add_node(false, None, None, &mut state);
        let e1 = g.add_edge(n1, n3, ClauseIndex(0), &mut state);
        let e2 = g.add_edge(n2, n3, ClauseIndex(0), &mut state);

        let n1_outgoing_edges: Vec<EdgeIndex> = g.outgoings(n1).collect();
        let n1_incoming_edges: Vec<EdgeIndex> = g.incomings(n1).collect();

        assert_eq!(vec![e1], n1_outgoing_edges);
        assert_eq!(0, n1_incoming_edges.len());

        let n2_outgoing_edges: Vec<EdgeIndex> = g.outgoings(n2).collect();
        let n2_incoming_edges: Vec<EdgeIndex> = g.incomings(n2).collect();

        assert_eq!(vec![e2], n2_outgoing_edges);
        assert_eq!(0, n2_incoming_edges.len());

        let n3_outgoing_edges: Vec<EdgeIndex> = g.outgoings(n3).collect();
        let n3_incoming_edges: Vec<EdgeIndex> = g.incomings(n3).collect();

        assert_eq!(0, n3_outgoing_edges.len());
        assert_eq!(vec![e2, e1], n3_incoming_edges);
    }

    #[test]
    fn add_multiple_outgoing_edges_to_nodes() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        let n1 = g.add_node(false, None, None, &mut state);
        let n2 = g.add_distribution(&vec![0.2], &mut state)[0];
        let n3 = g.add_node(false, None, None, &mut state);
        let n4 = g.add_distribution(&vec![0.3], &mut state)[0];

        let e1 = g.add_edge(n1, n2, ClauseIndex(0), &mut state);
        let e2 = g.add_edge(n1, n3, ClauseIndex(0), &mut state);
        let e3 = g.add_edge(n1, n4, ClauseIndex(0), &mut state);

        let outgoings: Vec<EdgeIndex> = g.outgoings(n1).collect();

        assert_eq!(vec![e3, e2, e1], outgoings);
    }

    #[test]
    fn setting_nodes_values() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        let n1 = g.add_node(false, None, None, &mut state);
        assert!(!g.is_node_bound(n1, &state));

        state.save_state();

        g.set_node(n1, true, &mut state);
        assert!(g.is_node_bound(n1, &state));
        assert!(g.nodes[n1.0].value);

        state.restore_state();
        assert!(!g.is_node_bound(n1, &state));
    }

    #[test]
    fn incremental_computation_of_objective() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        assert_eq!(0.0, g.get_objective(&state));
        let n1 = g.add_node(false, None, None, &mut state);
        let n2 = g.add_node(true, Some(0.4), Some(DistributionIndex(0)), &mut state);
        let n3 = g.add_node(true, Some(0.6), Some(DistributionIndex(0)), &mut state);
        assert_eq!(0.0, g.get_objective(&state));

        state.save_state();

        g.set_node(n1, true, &mut state);
        assert_eq!(0.0, g.get_objective(&state));

        g.set_node(n2, true, &mut state);
        assert_eq!(0.4, g.get_objective(&state));

        state.restore_state();

        assert_eq!(0.0, g.get_objective(&state));

        state.save_state();

        g.set_node(n2, true, &mut state);
        g.set_node(n3, true, &mut state);
        assert_eq!(1.0, g.get_objective(&state));
    }

    #[test]
    fn graph_add_distribution() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        assert_eq!(0, g.distributions.len());
        let weights = vec![0.2, 0.4, 0.1, 0.3];
        let nodes = g.add_distribution(&weights, &mut state);
        assert_eq!(
            vec![NodeIndex(0), NodeIndex(1), NodeIndex(2), NodeIndex(3)],
            nodes
        );
        assert_eq!(1, g.distributions.len());
        let distribution = g.distributions[0];
        assert_eq!(NodeIndex(0), distribution.first);
        assert_eq!(4, distribution.size);
        assert_eq!(0, state.get_int(distribution.active_edges));
        assert_eq!(0, state.get_int(distribution.nodes_false));
    }

    #[test]
    fn graph_add_multiple_distributions() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        let w1 = vec![0.1, 0.3, 0.6];
        let w2 = vec![0.5, 0.4, 0.05, 0.05];
        let w3 = vec![0.9, 0.1];

        let n1 = g.add_distribution(&w1, &mut state);
        assert_eq!(vec![NodeIndex(0), NodeIndex(1), NodeIndex(2)], n1);
        assert_eq!(1, g.distributions.len());
        let distribution = g.distributions[0];
        assert_eq!(n1[0], distribution.first);
        assert_eq!(3, distribution.size);
        assert_eq!(0, state.get_int(distribution.active_edges));
        assert_eq!(0, state.get_int(distribution.nodes_false));

        g.add_node(false, None, None, &mut state);
        let n2 = g.add_distribution(&w2, &mut state);
        assert_eq!(
            vec![NodeIndex(4), NodeIndex(5), NodeIndex(6), NodeIndex(7)],
            n2
        );
        assert_eq!(2, g.distributions.len());
        let distribution = g.distributions[1];
        assert_eq!(n2[0], distribution.first);
        assert_eq!(4, distribution.size);
        assert_eq!(0, state.get_int(distribution.active_edges));
        assert_eq!(0, state.get_int(distribution.nodes_false));

        g.add_node(false, None, None, &mut state);
        g.add_node(false, None, None, &mut state);

        let n3 = g.add_distribution(&w3, &mut state);
        assert_eq!(vec![NodeIndex(10), NodeIndex(11)], n3);
        assert_eq!(3, g.distributions.len());
        let distribution = g.distributions[2];
        assert_eq!(n3[0], distribution.first);
        assert_eq!(2, distribution.size);
        assert_eq!(0, state.get_int(distribution.active_edges));
        assert_eq!(0, state.get_int(distribution.nodes_false));
    }

    #[test]
    fn graph_add_edges_increases_edge_counts() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        let n1 = g.add_node(false, None, None, &mut state);
        let n2 = g.add_node(false, None, None, &mut state);
        let n3 = g.add_node(false, None, None, &mut state);
        let n4 = g.add_node(false, None, None, &mut state);

        g.clauses.push(Clause {
            first: EdgeIndex(0),
            size: 4,
            head: NodeIndex(0),
            active_edges: state.manage_int(4),
        });

        assert_eq!(0, g.node_number_incoming(n1, &state));
        assert_eq!(0, g.node_number_outgoing(n1, &state));
        assert_eq!(0, g.node_number_incoming(n2, &state));
        assert_eq!(0, g.node_number_outgoing(n2, &state));
        assert_eq!(0, g.node_number_incoming(n3, &state));
        assert_eq!(0, g.node_number_outgoing(n3, &state));
        assert_eq!(0, g.node_number_incoming(n4, &state));
        assert_eq!(0, g.node_number_outgoing(n4, &state));

        let e1 = g.add_edge(n1, n2, ClauseIndex(0), &mut state);
        assert_eq!(0, g.node_number_incoming(n1, &state));
        assert_eq!(1, g.node_number_outgoing(n1, &state));
        assert_eq!(1, g.node_number_incoming(n2, &state));
        assert_eq!(0, g.node_number_outgoing(n2, &state));
        assert_eq!(0, g.node_number_incoming(n3, &state));
        assert_eq!(0, g.node_number_outgoing(n3, &state));
        assert_eq!(0, g.node_number_incoming(n4, &state));
        assert_eq!(0, g.node_number_outgoing(n4, &state));

        let e2 = g.add_edge(n1, n3, ClauseIndex(0), &mut state);
        assert_eq!(0, g.node_number_incoming(n1, &state));
        assert_eq!(2, g.node_number_outgoing(n1, &state));
        assert_eq!(1, g.node_number_incoming(n2, &state));
        assert_eq!(0, g.node_number_outgoing(n2, &state));
        assert_eq!(1, g.node_number_incoming(n3, &state));
        assert_eq!(0, g.node_number_outgoing(n3, &state));
        assert_eq!(0, g.node_number_incoming(n4, &state));
        assert_eq!(0, g.node_number_outgoing(n4, &state));

        let e3 = g.add_edge(n4, n1, ClauseIndex(0), &mut state);
        assert_eq!(1, g.node_number_incoming(n1, &state));
        assert_eq!(2, g.node_number_outgoing(n1, &state));
        assert_eq!(1, g.node_number_incoming(n2, &state));
        assert_eq!(0, g.node_number_outgoing(n2, &state));
        assert_eq!(1, g.node_number_incoming(n3, &state));
        assert_eq!(0, g.node_number_outgoing(n3, &state));
        assert_eq!(0, g.node_number_incoming(n4, &state));
        assert_eq!(1, g.node_number_outgoing(n4, &state));

        let e4 = g.add_edge(n2, n4, ClauseIndex(0), &mut state);
        assert_eq!(1, g.node_number_incoming(n1, &state));
        assert_eq!(2, g.node_number_outgoing(n1, &state));
        assert_eq!(1, g.node_number_incoming(n2, &state));
        assert_eq!(1, g.node_number_outgoing(n2, &state));
        assert_eq!(1, g.node_number_incoming(n3, &state));
        assert_eq!(0, g.node_number_outgoing(n3, &state));
        assert_eq!(1, g.node_number_incoming(n4, &state));
        assert_eq!(1, g.node_number_outgoing(n4, &state));

        g.deactivate_edge(e1, &mut state);
        assert_eq!(1, g.node_number_incoming(n1, &state));
        assert_eq!(1, g.node_number_outgoing(n1, &state));
        assert_eq!(0, g.node_number_incoming(n2, &state));
        assert_eq!(1, g.node_number_outgoing(n2, &state));
        assert_eq!(1, g.node_number_incoming(n3, &state));
        assert_eq!(0, g.node_number_outgoing(n3, &state));
        assert_eq!(1, g.node_number_incoming(n4, &state));
        assert_eq!(1, g.node_number_outgoing(n4, &state));

        g.deactivate_edge(e2, &mut state);
        assert_eq!(1, g.node_number_incoming(n1, &state));
        assert_eq!(0, g.node_number_outgoing(n1, &state));
        assert_eq!(0, g.node_number_incoming(n2, &state));
        assert_eq!(1, g.node_number_outgoing(n2, &state));
        assert_eq!(0, g.node_number_incoming(n3, &state));
        assert_eq!(0, g.node_number_outgoing(n3, &state));
        assert_eq!(1, g.node_number_incoming(n4, &state));
        assert_eq!(1, g.node_number_outgoing(n4, &state));

        g.deactivate_edge(e3, &mut state);
        assert_eq!(0, g.node_number_incoming(n1, &state));
        assert_eq!(0, g.node_number_outgoing(n1, &state));
        assert_eq!(0, g.node_number_incoming(n2, &state));
        assert_eq!(1, g.node_number_outgoing(n2, &state));
        assert_eq!(0, g.node_number_incoming(n3, &state));
        assert_eq!(0, g.node_number_outgoing(n3, &state));
        assert_eq!(1, g.node_number_incoming(n4, &state));
        assert_eq!(0, g.node_number_outgoing(n4, &state));

        g.deactivate_edge(e4, &mut state);
        assert_eq!(0, g.node_number_incoming(n1, &state));
        assert_eq!(0, g.node_number_outgoing(n1, &state));
        assert_eq!(0, g.node_number_incoming(n2, &state));
        assert_eq!(0, g.node_number_outgoing(n2, &state));
        assert_eq!(0, g.node_number_incoming(n3, &state));
        assert_eq!(0, g.node_number_outgoing(n3, &state));
        assert_eq!(0, g.node_number_incoming(n4, &state));
        assert_eq!(0, g.node_number_outgoing(n4, &state));
    }

    #[test]
    fn add_clause_correctly_add_edges() {
        let mut state = TrailedStateManager::new();
        let mut g: Graph = Graph::new(&mut state);
        let n1 = g.add_node(false, None, None, &mut state);
        let n2 = g.add_node(false, None, None, &mut state);
        let n3 = g.add_node(false, None, None, &mut state);
        let n4 = g.add_node(false, None, None, &mut state);

        // Clause n2 && n3 => n1
        let c1 = g.add_clause(n1, &vec![n2, n3], &mut state);
        assert_eq!(ClauseIndex(0), c1);
        let first = g.clauses[c1.0].first.0;
        let s = g.clauses[c1.0].size;
        let sources = vec![n2, n3];
        for i in first..(first + s) {
            let edge = &g.edges[i];
            assert_eq!(sources[i - first], edge.src);
            assert_eq!(n1, edge.dst);
            assert_eq!(c1, edge.clause);
        }

        // Clause n3 && n4 => n2
        let c2 = g.add_clause(n2, &vec![n3, n4], &mut state);
        assert_eq!(ClauseIndex(1), c2);
        let first = g.clauses[c2.0].first.0;
        let s = g.clauses[c2.0].size;
        let sources = vec![n3, n4];
        for i in first..(first + s) {
            let edge = &g.edges[i];
            assert_eq!(sources[i - first], edge.src);
            assert_eq!(n2, edge.dst);
            assert_eq!(c2, edge.clause);
        }
    }

    #[test]
    fn clauses_are_correctly_collected() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        let n1 = g.add_node(false, None, None, &mut state);
        let n2 = g.add_node(false, None, None, &mut state);
        let n3 = g.add_node(false, None, None, &mut state);
        let n4 = g.add_node(false, None, None, &mut state);

        let empty: Vec<ClauseIndex> = vec![];
        // n1 && n2 => n3
        let c1 = g.add_clause(n3, &vec![n1, n2], &mut state);
        let n1_clauses: Vec<ClauseIndex> = g.node_clauses(n1).collect();
        let n2_clauses: Vec<ClauseIndex> = g.node_clauses(n2).collect();
        let n3_clauses: Vec<ClauseIndex> = g.node_clauses(n3).collect();
        let n4_clauses: Vec<ClauseIndex> = g.node_clauses(n4).collect();
        assert_eq!(vec![c1], n1_clauses);
        assert_eq!(vec![c1], n2_clauses);
        assert_eq!(vec![c1], n3_clauses);
        assert_eq!(empty, n4_clauses);

        // n2 && n4 => n1
        let c2 = g.add_clause(n1, &vec![n2, n4], &mut state);
        let n1_clauses: Vec<ClauseIndex> = g.node_clauses(n1).collect();
        let n2_clauses: Vec<ClauseIndex> = g.node_clauses(n2).collect();
        let n3_clauses: Vec<ClauseIndex> = g.node_clauses(n3).collect();
        let n4_clauses: Vec<ClauseIndex> = g.node_clauses(n4).collect();
        assert_eq!(vec![c2, c1], n1_clauses);
        assert_eq!(vec![c2, c1], n2_clauses);
        assert_eq!(vec![c1], n3_clauses);
        assert_eq!(vec![c2], n4_clauses);

        // n3 && n1 => n2
        let c3 = g.add_clause(n2, &vec![n3, n1], &mut state);
        let n1_clauses: Vec<ClauseIndex> = g.node_clauses(n1).collect();
        let n2_clauses: Vec<ClauseIndex> = g.node_clauses(n2).collect();
        let n3_clauses: Vec<ClauseIndex> = g.node_clauses(n3).collect();
        let n4_clauses: Vec<ClauseIndex> = g.node_clauses(n4).collect();
        assert_eq!(vec![c2, c3, c1], n1_clauses);
        assert_eq!(vec![c3, c2, c1], n2_clauses);
        assert_eq!(vec![c1, c3], n3_clauses);
        assert_eq!(vec![c2], n4_clauses);
    }
}

#[cfg(test)]
mod test_graph_iterators {

    use crate::core::graph::*;

    #[test]
    fn nodes_iterator() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        let nodes: Vec<NodeIndex> = (0..10)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect();
        let iterated: Vec<NodeIndex> = g.nodes_iter().collect();
        assert_eq!(nodes, iterated);
    }

    #[test]
    fn distributions_iterator() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        let distributions: Vec<Vec<NodeIndex>> = (0..5)
            .map(|_| g.add_distribution(&vec![0.3, 0.5, 0.2], &mut state))
            .collect();
        for d in 0..5 {
            let iterated: Vec<NodeIndex> = g.distribution_iter(DistributionIndex(d)).collect();
            assert_eq!(distributions[d], iterated);
            for n in 0..3 {
                let iterated_from_nodes: Vec<NodeIndex> =
                    g.nodes_distribution_iter(NodeIndex(d * 3 + n)).collect();
                assert_eq!(distributions[d], iterated_from_nodes);
            }
        }
    }

    #[test]
    fn clause_iterator() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
        let n: Vec<NodeIndex> = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect();
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3]], &mut state);
        g.add_clause(n[3], &vec![n[4]], &mut state);

        let clauses: Vec<Vec<ClauseIndex>> = (0..5)
            .map(|i| g.node_clauses(NodeIndex(i)).collect())
            .collect();
        assert_eq!(vec![ClauseIndex(0)], clauses[0]);
        assert_eq!(vec![ClauseIndex(1), ClauseIndex(0)], clauses[1]);
        assert_eq!(vec![ClauseIndex(0)], clauses[2]);
        assert_eq!(vec![ClauseIndex(2), ClauseIndex(1)], clauses[3]);
        assert_eq!(vec![ClauseIndex(2)], clauses[4]);
    }
}
