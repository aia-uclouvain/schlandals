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
//!
//! # Note:
//!     Once the graph is constructed, no edge/node should be removed from it. Thus this
//!     implementation does not have problems like dangling indexes.

#![allow(dead_code)]
use std::ops::{Index, IndexMut};

/// This is an abstraction that represents a node index. It is used to retrieve the `NodeData` in
/// the `Graph` representation
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct NodeIndex(pub usize);

/// This is an abstraction that represents an edge index. It is used to retrieve the `EdgeData` in
/// the `Graph` representation
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct EdgeIndex(pub usize);

#[derive(Debug, Copy, Clone)]
pub enum NodeValue {
    False = 0,
    True = 1,
    Unconstrained = 2,
}

/// Data structure that actually holds the data of a  node in the graph
#[derive(Debug, Copy, Clone)]
pub struct NodeData {
    /// The value associated to the node. This represent the assignment to the literal represented
    /// by the node.
    /// If the node is unassigned, this is None. Otherwise it is either True, False or
    /// Unconstrained if the node do not constraint the model count.
    pub value: Option<NodeValue>,

    /// Indicate if the literal represented by the node is a probabilistic literal (i.e. have a
    /// weight) or not
    pub probabilistic: bool,

    /// If `probabilistic` is `true`, then this is the weight associated to the node. Otherwise
    /// this is None
    pub weight: Option<f64>,

    /// If present, the index of the first outgoing edge of the node
    pub children: Option<EdgeIndex>,
    /// If present, the index of the first incoming edge of the node
    pub parents: Option<EdgeIndex>,
}

impl NodeData {
    pub fn new(probabilistic: bool, weight: Option<f64>) -> Self {
        Self {
            value: None,
            probabilistic,
            weight,
            children: None,
            parents: None,
        }
    }

    pub fn is_probabilistic(&self) -> bool {
        self.probabilistic
    }
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
}

impl EdgeData {
    pub fn new(
        src: NodeIndex,
        dst: NodeIndex,
        next_incoming: Option<EdgeIndex>,
        next_outgoing: Option<EdgeIndex>,
    ) -> Self {
        Self {
            src,
            dst,
            next_incoming,
            next_outgoing,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<NodeData>,
    pub edges: Vec<EdgeData>,
}

impl Index<NodeIndex> for Graph {
    type Output = NodeData;
    fn index(&self, idx: NodeIndex) -> &NodeData {
        &self.nodes[idx.0]
    }
}

impl IndexMut<NodeIndex> for Graph {
    fn index_mut(&mut self, idx: NodeIndex) -> &mut NodeData {
        &mut self.nodes[idx.0]
    }
}

impl Index<EdgeIndex> for Graph {
    type Output = EdgeData;
    fn index(&self, idx: EdgeIndex) -> &EdgeData {
        &self.edges[idx.0]
    }
}

impl IndexMut<EdgeIndex> for Graph {
    fn index_mut(&mut self, idx: EdgeIndex) -> &mut EdgeData {
        &mut self.edges[idx.0]
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            edges: vec![],
        }
    }

    pub fn nb_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn nb_edges(&self) -> usize {
        self.edges.len()
    }

    /// Add a node to the graph. At the creation of the graph, the nodes do not have any value. If
    /// the node represent a probabilistic literal (`probabilistic = true`), then it has a weight
    /// of `weight`. This method returns the index of the node.
    pub fn add_node(&mut self, probabilistic: bool, weight: Option<f64>) -> NodeIndex {
        let id = self.nodes.len();
        self.nodes.push(NodeData {
            value: None,
            probabilistic,
            weight,
            children: None,
            parents: None,
        });
        NodeIndex(id)
    }

    /// Add an edge between the node identified by `src` to the node identified by `dst`. This
    /// method returns the index of the edge.
    pub fn add_edge(&mut self, src: NodeIndex, dst: NodeIndex) -> EdgeIndex {
        let id = self.edges.len();
        let source_outgoing = self.nodes[src.0].children;
        let dest_incoming = self.nodes[dst.0].parents;
        let edge = EdgeData {
            src,
            dst,
            next_incoming: dest_incoming,
            next_outgoing: source_outgoing,
        };
        let index = EdgeIndex(id);
        self.nodes[src.0].children = Some(index);
        self.nodes[dst.0].parents = Some(index);
        self.edges.push(edge);
        index
    }

    /// Returns an iterator over the nodes that points to `target`
    pub fn parents(&self, target: NodeIndex) -> Parents<'_> {
        let first = self.nodes[target.0].parents;
        Parents {
            graph: self,
            next: first,
        }
    }

    /// Returns an iterator over the nodes pointed to by `source`
    pub fn children(&self, source: NodeIndex) -> Children<'_> {
        let first = self.nodes[source.0].children;
        Children {
            graph: self,
            next: first,
        }
    }
}

pub struct Parents<'g> {
    graph: &'g Graph,
    next: Option<EdgeIndex>,
}

impl<'g> Iterator for Parents<'g> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next {
            None => None,
            Some(eidx) => {
                let edge = &self.graph.edges[eidx.0];
                self.next = edge.next_incoming;
                Some(edge.src)
            }
        }
    }
}

pub struct Children<'g> {
    graph: &'g Graph,
    next: Option<EdgeIndex>,
}

impl<'g> Iterator for Children<'g> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next {
            None => None,
            Some(eidx) => {
                let edge = &self.graph.edges[eidx.0];
                self.next = edge.next_outgoing;
                Some(edge.dst)
            }
        }
    }
}

#[cfg(test)]
mod test_node_data {
    use crate::core::graph::NodeData;

    #[test]
    fn new_can_create_probabilistic_node() {
        let node = NodeData::new(true, Some(0.4));
        assert_eq!(true, node.is_probabilistic());
        assert!(node.weight.is_some());
        assert_eq!(0.4, node.weight.unwrap());
    }

    #[test]
    fn new_can_create_deterministic_node() {
        let node = NodeData::new(false, None);
        assert_eq!(false, node.is_probabilistic());
        assert_eq!(None, node.weight);
    }
}

#[cfg(test)]
mod test_edge_data {
    use crate::core::graph::{EdgeData, EdgeIndex, NodeIndex};

    #[test]
    fn new_can_create_first() {
        let src = NodeIndex(11);
        let dst = NodeIndex(13);
        let edge = EdgeData::new(src, dst, None, None);
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
        let edge = EdgeData::new(src, dst, next_incoming, next_outgoing);
        assert_eq!(next_incoming, edge.next_incoming);
        assert_eq!(next_outgoing, edge.next_outgoing);
    }
}

#[cfg(test)]
mod test_graph {
    use crate::core::graph::{EdgeIndex, Graph, NodeIndex};

    #[test]
    fn new_create_empty_graph() {
        let g = Graph::new();
        assert_eq!(0, g.nb_nodes());
        assert_eq!(0, g.nb_edges());
    }

    #[test]
    fn add_nodes_increment_index() {
        let mut g = Graph::new();
        for i in 0..10 {
            let idx = g.add_node(false, None);
            assert_eq!(NodeIndex(i), idx);
        }
    }

    #[test]
    fn add_edges_increment_index() {
        let mut g = Graph::new();
        g.add_node(false, None);
        g.add_node(false, None);
        for i in 0..10 {
            let idx = g.add_edge(NodeIndex(0), NodeIndex(1));
            assert_eq!(EdgeIndex(i), idx);
        }
    }

    #[test]
    fn add_multiple_incoming_edges_to_node() {
        let mut g = Graph::new();
        let n1 = g.add_node(false, None);
        let n2 = g.add_node(false, None);
        let n3 = g.add_node(false, None);
        g.add_edge(n1, n3);
        g.add_edge(n2, n3);

        let n1_outgoing: Vec<NodeIndex> = g.children(n1).collect();
        let n1_incoming: Vec<NodeIndex> = g.parents(n1).collect();

        assert_eq!(vec![n3], n1_outgoing);
        assert_eq!(0, n1_incoming.len());

        let n2_outgoing: Vec<NodeIndex> = g.children(n2).collect();
        let n2_incoming: Vec<NodeIndex> = g.parents(n2).collect();

        assert_eq!(vec![n3], n2_outgoing);
        assert_eq!(0, n2_incoming.len());

        let n3_outgoing: Vec<NodeIndex> = g.children(n3).collect();
        let n3_incoming: Vec<NodeIndex> = g.parents(n3).collect();

        assert_eq!(0, n3_outgoing.len());
        assert_eq!(vec![n2, n1], n3_incoming);
    }

    #[test]
    fn add_multiple_outgoing_edges_to_nodes() {
        let mut g = Graph::new();
        let n1 = g.add_node(false, None);
        let n2 = g.add_node(true, Some(0.2));
        let n3 = g.add_node(false, None);
        let n4 = g.add_node(true, Some(0.3));

        g.add_edge(n1, n2);
        g.add_edge(n1, n3);
        g.add_edge(n1, n4);

        let children: Vec<NodeIndex> = g.children(n1).collect();
        assert_eq!(vec![n4, n3, n2], children);
    }
}
