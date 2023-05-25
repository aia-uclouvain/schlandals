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

// This file is present in this repo for the time being, but ultimately I would like all
// AOMDD related file to be in their own crate

///! This module provide an implemantation of a weighted AND/OR MDD as defined by Robert Mateescu and Rina Dechter,
///! ´´AND/OR Multi-Valued Decision Diagrams (AOMDDs) for Weighted Graphical Models''

use crate::core::graph::{DistributionIndex, VariableIndex};
use rug::Float;
use crate::common::f128;

/// The identifier of a or-node in the ´or_nodes´ vector of the AOMDD
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct OrNodeIndex(usize);

/// The identifier of a and-node in the ´and_nodes´ vector of the AOMDD
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AndNodeIndex(usize);

/// The identifier of a node in the ´nodes´ vector of the AOMDD
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NodeIndex(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct OrAndArcIndex(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AndOrArcIndex(usize);

/// Structure representing the compiled AND/OR Multi-valued decision diagram (AOMDD).
pub struct AOMDD {
    /// Vector containing all the or-nodes of the AOMDD
    or_nodes: Vec<OrNode>,
    /// Vector containing all the and-nodes of the AOMDD
    and_nodes: Vec<AndNode>,
    /// Vector containing all or-and arcs
    or_and_arcs: Vec<OrAndArc>,
    /// Root of the AOMDD
    root: OrNodeIndex,
    /// Weight of the initial propagation
    weight_factor: Float,
}

/// Structure representing an OR node in the AOMDD. This node hold the decision variable to branch on.
/// Its children are the AND nodes corresponding the valid assignement on the decision.
pub struct OrNode {
    /// The decision of the OR node (i.e. which distribution to branch on)
    decision: DistributionIndex,
    /// Vectors containing the pointer to all AND nodes children, with the weight associated to the arc
    out_arcs: Vec<OrAndArcIndex>,
}

/// Structure representing an AND node. This node holds the assignment on its parent (OR node) decision.
/// It has a children OR node for the following decision and has as many children as independent component in
/// the sub-problem.
pub struct AndNode {
    /// Value assigned to the decision of the parent OR node
    assignment: VariableIndex,
    /// Children of the AND node. One children per independent sub-problem after all the propagations have been done
    /// on the assignment from the root until this node.
    children: Vec<OrNodeIndex>,
}

/// Structure representing an arc from an OR node to an AND node.
pub struct OrAndArc {
    /// The target of the arc
    to: AndNodeIndex,
    /// Weight of the arc
    weight: Float,
}

impl AOMDD {

    pub fn new() -> Self {
        let mut aomdd = Self {
            or_nodes: vec![
                OrNode {
                    decision: DistributionIndex(0),
                    out_arcs: vec![],
                },
                OrNode {
                    decision: DistributionIndex(0),
                    out_arcs: vec![],
                },                
            ],
            and_nodes: vec![],
            or_and_arcs: vec![],
            root: OrNodeIndex(0),
            weight_factor: f128!(1.0),
        };
        // Ensuring the root is consistent by default. I might change the node index of the consistent node
        // in the future and forgot it was hardcoded at the creation of the data structure. If any change is
        // made to the ´get_terminal_consistent´, this will be correctly updated
        aomdd.set_consistent_root();        
        aomdd
    }
    
    /// Adds an OR node to the diagram and returns its index.
    pub fn add_or_node(&mut self, decision: DistributionIndex) -> OrNodeIndex {
        let id = OrNodeIndex(self.or_nodes.len());
        self.or_nodes.push(OrNode {
            decision,
            out_arcs: vec![],
        });
        id
    }
    
    /// Adds an AND node to the given OR node
    pub fn add_and_node(&mut self, assignment: VariableIndex, parent: OrNodeIndex, weight: Float) -> AndNodeIndex {
        let id = AndNodeIndex(self.and_nodes.len());
        self.and_nodes.push(AndNode {
            assignment,
            children: vec![],
        });
        self.add_or_and_arc(parent, id, weight);
        id
    }
    
    fn add_or_and_arc(&mut self, from: OrNodeIndex, to: AndNodeIndex, weight: Float) -> OrAndArcIndex {
        let id = OrAndArcIndex(self.or_and_arcs.len());
        self.or_and_arcs.push(OrAndArc { to, weight });
        self.or_nodes[from.0].out_arcs.push(id);
        id
    }

    pub fn add_and_child(&mut self, parent: AndNodeIndex, child: OrNodeIndex) {
        self.and_nodes[parent.0].children.push(child);
    }
    
    /// Returns the decision of an OR node
    pub fn get_or_node_decision(&self, node: OrNodeIndex) -> DistributionIndex {
        self.or_nodes[node.0].decision
    }
    
    /// Returns the index of the consistent terminal node
    pub fn get_terminal_consistent(&self) -> OrNodeIndex {
        OrNodeIndex(0)
    }

    /// Returns the index of the inconsistent terminal node
    pub fn get_terminal_inconsistent(&self) -> OrNodeIndex {
        OrNodeIndex(1)
    }
    
    /// Sets the inconsistent terminal node as the root of the AOMDD
    pub fn set_inconsistent_root(&mut self) {
        self.root = self.get_terminal_inconsistent();
    }

    /// Sets the consistent terminal node as the root of the AOMDD
    pub fn set_consistent_root(&mut self) {
        self.root = self.get_terminal_consistent();
    }
    
    /// Sets the root of the AOMDDD to be the given node
    pub fn set_root(&mut self, node: OrNodeIndex) {
        self.root = node;
    }
    
    /// Sets the weight factor of the AOMDD
    pub fn set_weight_factor(&mut self, weight: Float) {
        self.weight_factor = weight;
    }
    
    fn evaluate_or(&self, node: OrNodeIndex) -> Float {
        if node == self.get_terminal_consistent() {
            f128!(1.0)
        } else if node == self.get_terminal_inconsistent() {
            f128!(0.0)
        } else {
            let mut p = f128!(0.0);
            for arc in self.or_nodes[node.0].out_arcs.iter().copied() {
                p += self.evaluate_and(self.or_and_arcs[arc.0].to) * &self.or_and_arcs[arc.0].weight;
            }
            p
        }
    }
    
    fn evaluate_and(&self, node: AndNodeIndex) -> Float {
        let mut p = f128!(1.0);
        for child in self.and_nodes[node.0].children.iter().copied() {
            p *= self.evaluate_or(child);
        }
        p
    }
    
    pub fn evaluate(&self) -> Float {
        self.evaluate_or(self.root) * &self.weight_factor
    }
    
    pub fn size(&self) -> usize {
        self.or_nodes.len() - 2
    }
}

// Various methods for dumping the AOMDD, including standardized format and graphviz (inspired from https://github.com/xgillard/ddo )

// Visualization as graphviz DOT file
impl AOMDD {
    
    pub fn as_graphviz(&self) -> String {
        
        let or_node_attributes = format!("shape=circle,style=filled");

        let mut out = String::new();
        
        out.push_str("digraph {\ntranksep = 3;\n\n");
        for (id, node) in self.or_nodes.iter().enumerate() {
            if id != self.get_terminal_consistent().0 && id != self.get_terminal_inconsistent().0 {
                let label = format!("d{}", node.decision.0);
                out.push_str(&self.node(id, &or_node_attributes, label));
                if self.get_terminal_consistent().0 != id && self.get_terminal_inconsistent().0 != id {
                    out.push_str(&self.meta_node(OrNodeIndex(id)));
                }

            }
        }
        
        for (id, node) in self.and_nodes.iter().enumerate() {
            let color = if node.children.len() == 1 {
                let child = node.children.first().unwrap();
                if *child == self.get_terminal_consistent() {
                    format!(",color=\"green\"")
                } else if *child == self.get_terminal_inconsistent() {
                    format!(",color=\"red\"")
                } else {
                    format!("")
                }
            } else {
                format!("")
            };
            let and_node_attributes = format!("shape=box,style=filled{}", color);
            out.push_str(&self.node(id + self.or_nodes.len(), &and_node_attributes, format!("v{}", node.assignment.0)));
            out.push_str(&self.edges_of_and(AndNodeIndex(id)));
        }
        
        out.push_str("}\n");
        out
    }
    
    fn node(&self, id: usize, attributes: &String, label: String) -> String {
        format!("\t{id} [{attributes},label=\"{label}\"];\n")
    }
    
    fn meta_node(&self, or_node: OrNodeIndex) -> String {
        let mut out = String::new();
        out.push_str(&format!("\tsubgraph cluster_meta_node_{} {{\n", or_node.0));
        let oid = or_node.0;
        for arc in self.or_nodes[or_node.0].out_arcs.iter().copied() {
            let aid = self.or_and_arcs[arc.0].to.0 + self.or_nodes.len();
            out.push_str(&format!("\t\t{} -> {};\n", oid, aid));
        }
        out.push_str("\t}\n");
        out
    }
    
    fn edge(from: usize, to: usize, cost: Option<Float>) -> String {
        let label = if let Some(f) = cost {
            format!("{:.4}", f)
        } else {
            format!("")
        };
        format!("\t{from} -> {to} [penwidth=1,label=\"{label}\"];\n")
    }

    fn edges_of_and(&self, node: AndNodeIndex) -> String {
        let mut out = String::new();
        for child in self.and_nodes[node.0].children.iter().copied() {
            if child != self.get_terminal_consistent() && child != self.get_terminal_inconsistent() {
                out.push_str(&Self::edge(
                    node.0 + self.or_nodes.len(),
                    child.0,
                    None,
                ))
            }
        }
        out
    }
}