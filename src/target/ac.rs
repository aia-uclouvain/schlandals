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
    /// Value propagated to true when branching (only fill if children of a sum node)
    propagated: Option<NodeIndex>,
    /// Edge to the first child
    first_child: Option<EdgeIndex>,
    /// Edge to the last child
    last_child: Option<EdgeIndex>,
    // True if the sub-circuit is complete
    complete: bool,
    /// Discrepancy of the node
    discrepancy: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum NodeType {
    Sum,
    Sub,
    Prod,
    Input,
}

#[derive(Default)]
pub struct Ac {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    cache: FxHashMap<Vec<usize>, NodeIndex>,
}

impl Ac {

    pub fn sum_node(&mut self) -> NodeIndex {
        self.nodes.push(Node {
            value: rational(0.0),
            nodetype: NodeType::Sum,
            distribution: None,
            propagated: None,
            first_child: None,
            last_child: None,
            complete: true,
            discrepancy: 0,
        });
        NodeIndex(self.nodes.len() - 1)
    }

    pub fn sub_node(&mut self) -> NodeIndex {
        self.nodes.push(Node {
            value: rational(0.0),
            nodetype: NodeType::Sub,
            distribution: None,
            propagated: None,
            first_child: None,
            last_child: None,
            complete: true,
            discrepancy: 0,
        });
        NodeIndex(self.nodes.len() - 1)
    }

    pub fn prod_node(&mut self) -> NodeIndex {
        self.nodes.push(Node {
            value: rational(1.0),
            nodetype: NodeType::Prod,
            distribution: None,
            propagated: None,
            first_child: None,
            last_child: None,
            complete: true,
            discrepancy: 0,
        });
        NodeIndex(self.nodes.len() - 1)
    }

    pub fn get_distribution_node(&mut self, distribution: DistributionIndex, variable: VariableIndex, value: Rational) -> NodeIndex {
        let key = vec![distribution.0, variable.0];
        match self.cache.get(&key) {
            Some(node) => *node,
            None => {
                let node = self.input_node(value);
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
            propagated: None,
            first_child: None,
            last_child: None,
            complete: true,
            discrepancy: 0,
        });
        NodeIndex(self.nodes.len() - 1)
    }

    pub fn add_edge(&mut self, parent: NodeIndex, child: NodeIndex) {
        // First, we update the value of the parent
        let child_value = match self[child].nodetype {
            NodeType::Sub => {
                if self[parent].first_child.is_none() {
                    -self[child].value.clone()
                } else {
                    self[child].value.clone()
                }
            },
            _ => self[child].value.clone(),
        };
        self.aggregate_child_value(parent, child_value);
        // Then, we add the edge
        let index = EdgeIndex(self.edges.len());
        self.edges.push(Edge {
            to: child,
            next: None,
        });
        if self[parent].first_child.is_none() {
            self[parent].first_child = Some(index);
            self[parent].last_child = Some(index);
        } else {
            let last_child = self[parent].last_child.unwrap();
            self[last_child].next = Some(index);
        }
    }

    pub fn number_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn number_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn set_propagated(&mut self, parent: NodeIndex, child: NodeIndex) {
        self[parent].propagated = Some(child);
    }

    fn aggregate_child_value(&mut self, node: NodeIndex, value: Rational) {
        match self[node].nodetype {
            NodeType::Sum => self[node].value += value,
            NodeType::Prod => self[node].value *= value,
            NodeType::Sub => self[node].value -= value,
            NodeType::Input => panic!("Doing arithmetic operation on input node"),
        }
    }

    pub fn evaluate(&mut self, node: NodeIndex) {
        let mut edge_ptr = self[node].first_child;
        let mut first_child = true;
        while let Some(edge) = edge_ptr {
            let child = self[edge].to;
            self.evaluate(child);
            let child_value = self[child].value.clone();
            match self[node].nodetype {
                NodeType::Sum => self[node].value += child_value,
                NodeType::Sub => {
                    if first_child {
                        self[node].value += child_value;
                        first_child = false;
                    } else {
                        self[node].value -= child_value;
                    }
                },
                NodeType::Prod => self[node].value *= child_value,
                NodeType::Input => panic!("Trying to do arithmetic operation on input nodes"),
            }
            edge_ptr = self[edge].next;
        }
        if let Some(child) = self[node].propagated {
            self.evaluate(child);
            let child_value = self[child].value.clone();
            self[node].value *= child_value;
        }
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

    pub fn value(&self) -> Rational {
        self.value.clone()
    }

    pub fn nodetype(&self) -> NodeType {
        self.nodetype
    }
}

impl Ac {

    pub fn to_graphviz(&self) -> String {
        let mut out = String::new();
        out.push_str("digraph {\ntranksep =3; \n\n");

        for node in (0..self.nodes.len()).map(NodeIndex) {
            let id = node.0;
            let value = format!("{:.4}", rational_to_f64(&self[node].value));
            match self[node].nodetype() {
                NodeType::Sum => {
                    out.push_str(&format!("\t{id} [shape=circle,style=filled,label=\"{id} | + | {value}\"];\n"));
                },
                NodeType::Sub => {
                    out.push_str(&format!("\t{id} [shape=circle,style=filled,label=\"{id} | - | {value}\"];\n"));
                },
                NodeType::Prod => {
                    out.push_str(&format!("\t{id} [shape=square,style=filled,label=\"{id} | * | {value}\"];\n"));
                },
                NodeType::Input => {
                    out.push_str(&format!("\t{id} [shape=doublecircle,style=filled,label=\"{id} | {value}\"];\n"));
                },
            }
        }

        for node in (0..self.nodes.len()).map(NodeIndex) {
            let mut edge_ptr = self[node].first_child;
            while let Some(edge) = edge_ptr {
                let child = self[edge].to;
                let from = node.0;
                let to = child.0;
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
