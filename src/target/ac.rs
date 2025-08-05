use super::{Edge, EdgeIndex, NodeIndex};
use crate::common::*;
use crate::core::problem::{DistributionIndex, VariableIndex};

use malachite::rational::Rational;
use rustc_hash::FxHashMap;

pub struct Node {
    value: Rational,
    /// Type of node
    nodetype: NodeType,
    /// Distribution to branch on if Sum node
    distribution: Option<DistributionIndex>,
    /// Edge to the first child
    first_parent: Option<EdgeIndex>,
    // True if the sub-circuit is complete
    complete: bool,
    /// True if the sub-circuit is SAT
    sat: bool,
    /// Discrepancy of the node
    discrepancy: usize,
    /// Position in the DAG (layer and position in the layer)
    position: (usize, usize),
}

#[derive(Clone, Copy, Debug)]
pub enum NodeType {
    Sum,
    Sub,
    Prod,
    Input,
}

pub struct Ac {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    cache: FxHashMap<Vec<usize>, NodeIndex>,
    layers: Vec<Vec<NodeIndex>>,
}

impl Default for Ac {
    fn default() -> Self {
        Self {
            nodes: vec![],
            edges: vec![],
            cache: FxHashMap::default(),
            layers: vec![vec![]],
        }
    }
}

impl Ac {

    pub fn sum_node(&mut self) -> NodeIndex {
        self.nodes.push(Node {
            value: rational(0.0),
            nodetype: NodeType::Sum,
            distribution: None,
            first_parent: None,
            complete: true,
            sat: true,
            discrepancy: 0,
            position: (0, 0),
        });
        NodeIndex(self.nodes.len() - 1)
    }

    pub fn sub_node(&mut self) -> NodeIndex {
        self.nodes.push(Node {
            value: rational(0.0),
            nodetype: NodeType::Sub,
            distribution: None,
            first_parent: None,
            complete: true,
            sat: true,
            discrepancy: 0,
            position: (0, 0),
        });
        NodeIndex(self.nodes.len() - 1)
    }

    pub fn prod_node(&mut self) -> NodeIndex {
        self.nodes.push(Node {
            value: rational(1.0),
            nodetype: NodeType::Prod,
            distribution: None,
            first_parent: None,
            complete: true,
            sat: true,
            discrepancy: 0,
            position: (0, 0),
        });
        NodeIndex(self.nodes.len() - 1)
    }

    pub fn get_distribution_node(&mut self, distribution: DistributionIndex, variable: VariableIndex, value: Rational) -> NodeIndex {
        let key = vec![distribution.0, variable.0];
        match self.cache.get(&key) {
            Some(node) => *node,
            None => {
                let node = self.input_node(value);
                let position_in_layer = self.layers[0].len();
                self[node].position = (0, position_in_layer);
                self.layers[0].push(node);
                self.cache.insert(key, node);
                node
            }
        }
    }

    pub fn input_node(&mut self, value: Rational) -> NodeIndex {
        self.nodes.push(Node {
            value,
            nodetype: NodeType::Input,
            distribution: None,
            first_parent: None,
            complete: true,
            sat: true,
            discrepancy: 0,
            position: (0, 0),
        });
        NodeIndex(self.nodes.len() - 1)
    }

    fn update_layer_positions(&mut self, node: NodeIndex, new_layer: usize) {
        if new_layer >= self.layers.len() {
            self.layers.push(vec![]);
        }
        let (layer, position_in_layer) = self[node].position;
        // New node, not in any layer
        if layer == 0 {
            self[node].position = (new_layer, self.layers[new_layer].len());
            self.layers[new_layer].push(node);
        } else if position_in_layer == self.layers[layer].len() - 1{
            // Last element, just removes it
            self.layers[layer].remove(position_in_layer);
            self[node].position = (new_layer, self.layers[new_layer].len());
            self.layers[new_layer].push(node);
        } else {
            // In the middle of a layer, we swap remove and update the swapped node
            self.layers[layer].swap_remove(position_in_layer);
            self[node].position = (new_layer, self.layers[new_layer].len());
            self.layers[new_layer].push(node);
            let node_to_update = self.layers[layer][position_in_layer];
            self[node_to_update].position = (layer, position_in_layer);
        }
    }

    fn remove_from_layer(&mut self, node: NodeIndex) {
        let (layer, position) = self[node].position;
        if self.layers[layer].len() == 1 {
            self.layers[layer].clear();
        } else {
            let to_swap = *self.layers[layer].last().unwrap();
            self.layers[layer].swap_remove(position);
            self[to_swap].position = (layer, position);
        }
    }

    pub fn add_edge(&mut self, parent: NodeIndex, child: NodeIndex) {
        let new_layer = self[child].position.0 + 1;
        if new_layer > self[parent].position.0 {
            debug_assert!(!matches!(self[parent].nodetype, NodeType::Input));
            self.update_layer_positions(parent, new_layer);
        }
        self.edges.push(Edge {
            to: parent,
            next: self[child].first_parent,
        });
        self[child].first_parent = Some(EdgeIndex(self.edges.len() - 1));
    }

    pub fn number_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn number_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn reset(&mut self) {
        for node in self.nodes.iter_mut() {
            match node.nodetype {
                NodeType::Sum => node.value = rational(0.0),
                NodeType::Sub => node.value = rational(0.0),
                NodeType::Prod => node.value = rational(1.0),
                NodeType::Input => (),
            }
        }
    }

    pub fn evaluate(&mut self) {
        self.reset();
        for layer in 0..self.layers.len() {
            for index in 0..self.layers[layer].len() {
                let node = self.layers[layer][index];
                let value = self[node].value.clone();
                let mut edge_ptr = self[node].first_parent;
                while let Some(edge) = edge_ptr {
                    let parent = self[edge].to;
                    match self[parent].nodetype {
                        NodeType::Sum => self[parent].value += &value,
                        NodeType::Sub => panic!("Sub node not yet implemented"),
                        NodeType::Prod => self[parent].value *= &value,
                        NodeType::Input => panic!("Trying to do arithmetic operation on input nodes"),
                    }
                    edge_ptr = self[edge].next;
                }
            }
        }
    }

    pub fn clean(&mut self) {
        let mut new_node_indexes: Vec<NodeIndex> = (0..self.nodes.len()).map(NodeIndex).collect();
        let mut size_nodes = self.nodes.len();
        for i in (0..self.nodes.len()).rev() {
            if !self.nodes[i].is_sat() {
                new_node_indexes.swap(i, size_nodes - 1);
                size_nodes -= 1;
                self.remove_from_layer(NodeIndex(i));
            }
        }

        let mut map_node = FxHashMap::<NodeIndex, NodeIndex>::default();
        for i in 0..size_nodes {
            let old_index = new_node_indexes[i];
            let new_index = NodeIndex(i);
            map_node.insert(old_index, new_index);
        }

        for layer in 0..self.layers.len() {
            for i in 0..self.layers[layer].len() {
                let node = self.layers[layer][i];
                self.layers[layer][i] = map_node[&node];
            }
        }

        let mut new_edge_indexes: Vec<EdgeIndex> = (0..self.edges.len()).map(EdgeIndex).collect();
        let mut size_edges = self.edges.len();
        for i in (0..self.edges.len()).rev() {
            let to = self.edges[i].to;
            if !map_node.contains_key(&to) {
                new_edge_indexes.swap(i, size_edges - 1);
                size_edges -= 1;
            }
        }

        let mut map_edges = FxHashMap::<EdgeIndex, EdgeIndex>::default();
        for i in 0..size_edges {
            let old_index = new_edge_indexes[i];
            let new_index = EdgeIndex(i);
            map_edges.insert(old_index, new_index);
        }

        for i in 0..size_nodes {
            let index = new_node_indexes[i].0;
            self.nodes.swap(i, index);
            let mut parent_ptr = self.nodes[i].first_parent;
            while parent_ptr.is_some() && !map_edges.contains_key(&parent_ptr.unwrap()) {
                let edge = parent_ptr.unwrap();
                parent_ptr = self[edge].next;
            }
            match parent_ptr {
                Some(e) => self.nodes[i].first_parent = Some(map_edges[&e]),
                None => self.nodes[i].first_parent = None,
            };
        }
        self.nodes.truncate(size_nodes);

        for i in 0..size_edges {
            let source = new_edge_indexes[i];
            let mut next = self[source].next;
            while next.is_some() && !map_edges.contains_key(&next.unwrap()) {
                next = self[next.unwrap()].next;
            }
            match next {
                Some(e) => self[source].next = Some(map_edges[&e]),
                None => self[source].next = None,
            };
            let old_to = self[source].to;
            self[source].to = map_node[&old_to];
        }
        for i in 0..size_edges {
            let index = new_edge_indexes[i].0;
            self.edges.swap(i, index);
        }
        self.edges.truncate(size_edges);

    }

}

impl Node {

    pub fn distribution(&self) -> Option<DistributionIndex> {
        self.distribution
    }

    pub fn set_distribution(&mut self, distribution: Option<DistributionIndex>) {
        self.distribution = distribution;
    }

    pub fn complete(&mut self) {
        self.complete = true;
    }

    pub fn incomplete(&mut self) {
        self.complete = false;
    }

    pub fn is_complete(&self) -> bool {
        self.complete
    }

    pub fn is_sat(&self) -> bool {
        self.sat
    }

    pub fn unsat(&mut self) {
        self.sat = false;
    }

    pub fn value(&self) -> Rational {
        self.value.clone()
    }

    pub fn nodetype(&self) -> NodeType {
        self.nodetype
    }

    pub fn discrepancy(&self) -> usize {
        self.discrepancy
    }

    pub fn set_discrepancy(&mut self, discrepancy: usize) {
        self.discrepancy = discrepancy;
    }

    pub fn first_parent(&self) -> Option<EdgeIndex> {
        self.first_parent
    }

}

impl Ac {

    pub fn to_graphviz(&self) -> String {
        let mut out = String::new();
        out.push_str("digraph {\ntranksep =3; \n\n");

        for node in (0..self.nodes.len()).map(NodeIndex) {
            let id = node.0;
            let value = format!("{:.4}", rational_to_f64(&self[node].value));
            let color = if self[node].is_sat() { "grey" } else { "red" };
            let layer = self[node].position.0;
            match self[node].nodetype() {
                NodeType::Sum => {
                    out.push_str(&format!("\t{id} [shape=circle,color={color},style=filled,layer={layer},label=\"{id} | + | {value}\"];\n"));
                },
                NodeType::Sub => {
                    out.push_str(&format!("\t{id} [shape=circle,color={color},style=filled,layer={layer},label=\"{id} | - | {value}\"];\n"));
                },
                NodeType::Prod => {
                    out.push_str(&format!("\t{id} [shape=square,color={color},style=filled,layer={layer},label=\"{id} | * | {value}\"];\n"));
                },
                NodeType::Input => {
                    out.push_str(&format!("\t{id} [shape=doublecircle,color={color},style=filled,layer={layer},label=\"{id} | {value}\"];\n"));
                },
            }
        }

        for node in (0..self.nodes.len()).map(NodeIndex) {
            let mut edge_ptr = self[node].first_parent;
            while let Some(edge) = edge_ptr {
                let parent = self[edge].to;
                let from = node.0;
                let to = parent.0;
                out.push_str(&format!("\t{from} -> {to} [penwidth=1];\n"));
                edge_ptr = self[edge].next;
            }
        }

        out.push_str("}\n");
        out
    }

}

impl std::ops::Index<NodeIndex> for Ac {
    type Output=Node;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl std::ops::IndexMut<NodeIndex> for Ac {

    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        &mut self.nodes[index.0]
    }
}

impl std::ops::Index<EdgeIndex> for Ac {
    type Output=Edge;

    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.edges[index.0]
    }
}

impl std::ops::IndexMut<EdgeIndex> for Ac {

    fn index_mut(&mut self, index: EdgeIndex) -> &mut Self::Output {
        &mut self.edges[index.0]
    }
}
