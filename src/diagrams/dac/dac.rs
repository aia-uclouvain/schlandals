//Schlandals
//Copyright (C) 2022 A. Dubray, L. Dierckx
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

//! This module provide an implementation of a distribution aware arithmetic circuits.
//! Unlike traditional circuits, the input of the circuits are not variables, but distributions
//! as used by Schlandal's modelling language.
//! Hence, given a valid input for Schlandals, there is one input per distribution.
//! Then, internal nodes are either a product node or a sum node. Once constructed, the probability
//! of the original problem can be computed in a feed-forward manner, starting from the input and pushing
//! the values towards the root of the circuits.
//! As of now, the circuit structure has been designed to be optimized when lots of queries are done on it.
//! A typical use case is when the parameter of the distributions must be learn in a EM like algorithm.

use std::{fmt, path::PathBuf, fs::File, io::{BufRead, BufReader}};
use rustc_hash::FxHashMap;

use rug::Float;
use crate::core::graph::{DistributionIndex, VariableIndex};
use crate::solvers::*;

use super::node::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex(pub usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LayerIndex(usize);

/// Structure representing the Distribution awared Arithmetic Circuit (DAC).
/// The structure only has three vector for the distributions, the internal nodes and the output of each internal node.
/// Currently the structure is reorganized after its creation to optimize the cache locality and simplify the evaluation.
/// That is, the nodes vector is sorted by "layer" so that all the inputs of node at index i are at an index j < i.
/// Moreover, the outputs of each internal nodes are stored in a contiguous part of the `outputs` vector.
/// This has two effects:
///     - When evaluating the circuits, a simple pass from index 0 until `nodes.len()` is sufficient. When loading node at index i,
///       the next node to be evaluated is i+1 and is likely to be stored in cache.
///     - When a node sends its value to its outputs, the same optimization happens (as loading the first output is likely to implies that
///       the following outputs are in cache).
pub struct Dac {
    /// Internal nodes of the circuit
    pub nodes: Vec<Node>,
    /// Outputs of the internal nodes
    outputs: Vec<NodeIndex>,
    /// Inputs of the internal nodes
    inputs: Vec<NodeIndex>,
    /// Mapping between the (distribution, value) and the node index for distribution nodes
    pub distribution_mapping: FxHashMap<(DistributionIndex, usize), NodeIndex>,
    /// An optional solver to evaluate nodes that have been cut due to partial compilation
    solver: Option<Solver>,
}

impl Dac {

    /// Creates a new empty DAC. An input node is created for each distribution in the graph.
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            outputs: vec![],
            inputs: vec![],
            distribution_mapping: FxHashMap::default(),
            solver: None,
        }
    }

    pub fn set_solver(&mut self, solver: Solver) {
        self.solver = Some(solver);
    }
    
    /// Adds a prod node to the circuit
    pub fn add_prod_node(&mut self) -> NodeIndex {
        let id = NodeIndex(self.nodes.len());
        self.nodes.push(Node::product());
        id
    }
    
    /// Adds a sum node to the circuit
    pub fn add_sum_node(&mut self) -> NodeIndex {
        let id = NodeIndex(self.nodes.len());
        self.nodes.push(Node::sum());
        id
    }

    /// Adds `output` to the outputs of `node` and `node` to the inputs of `output`. Note that this
    /// function uses the vectors in each node. They are transferred afterward in the `outputs` vector.
    pub fn add_node_output(&mut self, node: NodeIndex, output: NodeIndex) {
        self[node].add_output(output);
        self[output].add_input(node);
    }
    
    fn swap(&mut self, new: &mut [usize], old: &mut [usize], i: usize, j: usize) {
        self.nodes.swap(i, j);
        new[old[i]] = j;
        new[old[j]] = i;
        old.swap(i, j);
    }
    
    /// Layerizes the circuit. This function sort the internal nodes of the circuits such that an internal nodes at index i
    /// has all its input at an index j < i.
    pub fn layerize(&mut self) {
        // First, we need to add the layer to each node
        let mut to_process: Vec<(NodeIndex, usize)> = vec![];
        for node in (0..self.nodes.len()).map(NodeIndex) {
            if let TypeNode::Distribution {..} = self[node].get_type() {
                // The distribution nodes are at layer 0
                to_process.push((node, 0));
            }
            if self[node].is_node_incomplete() {
                to_process.push((node, 0));
            }
        }
        let mut number_layers = 1;
        while let Some((node, layer)) = to_process.pop() {
            // Only change if the layer must be increased
            if layer >= self[node].get_layer() {
                if number_layers < layer + 1 {
                    number_layers = layer + 1;
                }
                self[node].set_layer(layer);
                for output_id in self[node].iter_output() {
                    let output = self[node].get_output_at(output_id);
                    if self[output].get_layer() < layer + 1 {
                        to_process.push((output, layer+1));
                    }
                }
            }
        }

        let n_nodes = self.nodes.len();
        // new_indexes stores at each old index i the new index of the node.
        // The new index of a node with index i is new_indexes[i].
        let mut new_indexes = (0..n_nodes).collect::<Vec<usize>>();
        // old_indexes store for each each new index i the old index of the node.
        // This vector is used to be able to update new_indexes when nodes have been moved.
        let mut old_indexes = (0..n_nodes).collect::<Vec<usize>>();
        let mut end = self.nodes.len();
        let mut start = 0;
        // First we remove all nodes that are not part of a path from an input to the root of the
        // circuit.
        while start < end {
            if self.nodes[start].is_to_remove() {
                self.swap(&mut new_indexes, &mut old_indexes, start, end-1);
                end -= 1;
            } else {
                start += 1;
            }
        }

        // Then we process each layer one by one.
        let mut start_layer = 0;
        for layer in 0..number_layers {
            for i in start_layer..end{
                // If the node is part of the layer being process, then we swap it to the layer area
                let node = NodeIndex(i);
                if self[node].get_layer() == layer {
                    // Only move if necessary
                    if i != start_layer {
                        self.swap(&mut new_indexes, &mut old_indexes, start_layer, i);
                    }
                    start_layer += 1;
                }
            }
        }
        
        // At this point, the nodes vector is sorted by layer. But the indexes for the outputs/inputs must be updated.
        for node in (0..self.nodes.len()).map(NodeIndex) {
            // Drop all nodes that have been removed
            if let TypeNode::Distribution { d, v } = self[node].get_type() {
                self[node].retain_output_with_new_output(&new_indexes, end);
                self.distribution_mapping.insert((DistributionIndex(d), v), node);
            }

            if self[node].is_node_incomplete() {
                self[node].retain_output_with_new_output(&new_indexes, end);
            }

            // Update the outputs with the new indexes
            let n = self.outputs.len();
            self[node].set_output_start(n);
            self[node].set_number_outputs(0);

            for output_id in self[node].iter_output() {
                let output = self[node].get_output_at(output_id);
                // If the node has not been dropped, update the output. At this step, it is also moved in the
                // outputs vector of the DAC structure
                if new_indexes[output.0] < end {
                    let new_output = NodeIndex(new_indexes[output.0]);
                    self.outputs.push(new_output);
                    self[node].increment_number_output();
                }
            }
            self[node].clear_and_shrink_output();

            // Update the inputs
            let input_start = self[node].get_input_start();
            let number_input = self[node].get_number_inputs();

            for input_index in input_start..(input_start+number_input) {
                let old = self.inputs[input_index];
                self.inputs[input_index] = NodeIndex(new_indexes[old.0]);
            }
        }
        // Actually remove the nodes (and allocated space) from the nodes vector.
        self.nodes.truncate(end);
        self.nodes.shrink_to(end);
    }
    
    /// Tag all nodes that are not on a path from an input (incomplete nodes are considered as
    /// input) to the root of the DAC as to be removed.
    pub fn remove_dead_ends(&mut self) {
        let mut to_process: Vec<NodeIndex> = vec![];
        for node in (0..self.nodes.len()).map(NodeIndex) {
            if self[node].is_distribution() || self[node].is_node_incomplete() {
                for output_id in self[node].iter_output() {
                    let output = self[node].get_output_at(output_id);
                    self[node].set_to_remove(false);
                    to_process.push(output)
                }
            }
        }
        while let Some(node) = to_process.pop() {
            self[node].set_to_remove(false);
            for output_id in self[node].iter_output() {
                let output = self[node].get_output_at(output_id);
                if self[output].is_to_remove() {
                    to_process.push(output);
                }
            }
        }
    }
    
    /// A node is said to be neutral if it is not to be removed, has only one input, and is not the root of the DAC.
    /// In such case it can be bypass (it does not change its input)
    fn is_neutral(&self, node: NodeIndex) -> bool {
        !self[node].is_to_remove() && self[node].has_output() && self[node].get_number_inputs() == 1
    }
    
    /// Reduces the current circuit. This implies the following transformations (in that order)
    ///     1. Neutral nodes are tagged to be removed. A neutral node is a computation node (sum/prod nodes)
    ///        which i) are not the root node ii) have only one input. Such node do not modify their
    ///        input. Hence the output can be redirected to the node's output.
    ///     2. If a sum node has as input a distribution and all its value, it can be removed. In practice it only
    ///        has input from that distribution
    pub fn reduce(&mut self) {
        let mut changed = true;
        while changed {
            changed = false;
            // First, we remove all neutral node. Since it can lead to a possible optimization of the sum node, we do that first
            for node in (0..self.nodes.len()).map(NodeIndex) {
                if self.is_neutral(node) {
                    changed = true;
                    self[node].set_to_remove(true);
                    // The removal of the node is the same, no matter the node type of the input
                    let input = self[node].iter_input().next().unwrap();
                    // Removing the node from the output of the input node
                    if let Some(idx) = self[input].iter_output().position(|x| self[node].get_output_at(x) == node) {
                        self[input].remove_index_from_output(idx);
                    }
                    for output_id in self[node].iter_output() {
                        let output = self[node].get_output_at(output_id);
                        // Adds the input of node to its parent input and remove node from the inputs
                        self[input].add_output(output);
                        // Updating the input of the new output from node -> input
                        self[output].remove_input(node);
                        self[output].add_input(input);
                    }
                    self[node].clear_and_shrink_output();
                    self[node].clear_and_shrink_input();
                
                }
            }
                
            // If a distribution node send all its value to a sum node, remove the node from the output
            for node in (0..self.nodes.len()).map(NodeIndex) {
                if self[node].is_sum() {
                    let mut in_count = 0.0;
                    let mut distri_child: Option<usize> = None;
                    for input in self[node].iter_input() {
                        if let TypeNode::Distribution { d, v:_ } = self[input].get_type() {
                            if distri_child.is_none() {
                                distri_child = Some(d);
                                in_count += self[input].get_value().to_f64();
                            } else if Some(d) == distri_child {
                                in_count += self[node].get_value().to_f64();
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    if in_count == 1.0 {
                        changed = true;
                        for input in self[node].iter_input().collect::<Vec<NodeIndex>>() {
                            //debug_assert!(matches!(self.nodes[input.0].typenode, TypeNode::Distribution{..}));
                            let index_to_remove = self[input].iter_output().position(|x| self[node].get_output_at(x) == node).unwrap();
                            self[input].remove_index_from_output(index_to_remove);
                        }
                        self[node].set_to_remove(true);
                        self[node].clear_and_shrink_input();
                        for output_id in self[node].iter_output() {
                            let output = self[node].get_output_at(output_id);
                            self[output].remove_input(node);
                        }
                    }
                }
            }
        }

        // Move the inputs into the input vector
        for node in (0..self.nodes.len()).map(NodeIndex) {
            let n = self.inputs.len();
            self[node].set_input_start(n);
            for input in self[node].iter_input().collect::<Vec<NodeIndex>>() {
                self.inputs.push(input);
            }
            self[node].clear_and_shrink_input();
        }
    }
    
    // --- Evaluation ---- //

    /// Updates the values of the distributions
    pub fn reset_distributions(&mut self, distributions: &Vec<Vec<f64>>) {
        for node in (0..self.nodes.len()).map(NodeIndex) {
            if let TypeNode::Distribution { d, v } = self[node].get_type() {
                self[node].set_value(distributions[d][v]);
            }
            // All the distributions are at layer 0
            if self[node].get_layer() > 0 {
                break;
            }
        }
        if let Some(ref mut s) = self.solver {
            match s {
                Solver::SMinInDegree(ref mut solver) => solver.update_distributions(distributions),
                Solver::SMinOutDegree(ref mut solver) => solver.update_distributions(distributions),
                Solver::SMaxDegree(ref mut solver) => solver.update_distributions(distributions),
                Solver::SVSIDS(ref mut solver) => solver.update_distributions(distributions),
                Solver::QMinInDegree(ref mut solver) => solver.update_distributions(distributions),
                Solver::QMinOutDegree(ref mut solver) => solver.update_distributions(distributions),
                Solver::QMaxDegree(ref mut solver) => solver.update_distributions(distributions),
                Solver::QVSIDS(ref mut solver) => solver.update_distributions(distributions),
            }
        }
    }

    /// Sends the value of the node to its parents
    fn send_value(&mut self, node: NodeIndex, value: Float) {
        let start = self.nodes[node.0].get_output_start();
        let end = start + self.nodes[node.0].get_number_outputs();
        for i in start..end {
            let output = self.outputs[i];
            self[output].apply_value(&value);
        }
    }

    pub fn reset(&mut self) {
        for node in (0..self.nodes.len()).map(NodeIndex) {
            if self[node].is_node_incomplete() {
                let propagations = self[node].get_propagation().clone();
                let s = self.solver.as_mut().unwrap();
                match s {
                    Solver::SMinInDegree(ref mut solver) => {
                        let prefix_proba = solver.add_to_propagation_stack(&propagations);
                        match solver.solve() {
                            Ok(p) => {
                                self[node].set_value(p.to_f64() / prefix_proba.to_f64());
                            },
                            Err(_) => {println!("Error reset")}, // TODO what do we do if we can not evaluate the node ?
                        };
                    },
                    Solver::SMinOutDegree(ref mut solver) => {
                        let prefix_proba = solver.add_to_propagation_stack(&propagations);
                        match solver.solve() {
                            Ok(p) => {
                                self[node].set_value(p.to_f64() / prefix_proba.to_f64());
                            },
                            Err(_) => {println!("Error reset")}, // TODO what do we do if we can not evaluate the node ?
                        };
                    },
                    Solver::SMaxDegree(ref mut solver) => {
                        let prefix_proba = solver.add_to_propagation_stack(&propagations);
                        match solver.solve() {
                            Ok(p) => {
                                self[node].set_value(p.to_f64() / prefix_proba.to_f64());
                            },
                            Err(_) => {println!("Error reset")}, // TODO what do we do if we can not evaluate the node ?
                        };
                    },
                    Solver::SVSIDS(ref mut solver) => {
                        let prefix_proba = solver.add_to_propagation_stack(&propagations);
                        match solver.solve() {
                            Ok(p) => {
                                self[node].set_value(p.to_f64() / prefix_proba.to_f64());
                            },
                            Err(_) => {println!("Error reset")}, // TODO what do we do if we can not evaluate the node ?
                        };
                    },
                    Solver::QMinInDegree(ref mut solver) => {
                        let prefix_proba = solver.add_to_propagation_stack(&propagations);
                        match solver.solve() {
                            Ok(p) => {
                                self[node].set_value(p.to_f64() / prefix_proba.to_f64());
                            },
                            Err(_) => {println!("Error reset")}, // TODO what do we do if we can not evaluate the node ?
                        };
                    },
                    Solver::QMinOutDegree(ref mut solver) => {
                        let prefix_proba = solver.add_to_propagation_stack(&propagations);
                        match solver.solve() {
                            Ok(p) => {
                                self[node].set_value(p.to_f64() / prefix_proba.to_f64());
                            },
                            Err(_) => {println!("Error reset")}, // TODO what do we do if we can not evaluate the node ?
                        };
                    },
                    Solver::QMaxDegree(ref mut solver) => {
                        let prefix_proba = solver.add_to_propagation_stack(&propagations);
                        match solver.solve() {
                            Ok(p) => {
                                self[node].set_value(p.to_f64() / prefix_proba.to_f64());
                            },
                            Err(_) => {println!("Error reset")}, // TODO what do we do if we can not evaluate the node ?
                        };
                    },
                    Solver::QVSIDS(ref mut solver) => {
                        let prefix_proba = solver.add_to_propagation_stack(&propagations);
                        match solver.solve() {
                            Ok(p) => {
                                self[node].set_value(p.to_f64() / prefix_proba.to_f64());
                            },
                            Err(_) => {println!("Error reset")}, // TODO what do we do if we can not evaluate the node ?
                        };
                    },
                }
            } else {
                match self[node].get_type() {
                    TypeNode::Product => self[node].set_value(1.0),
                    TypeNode::Sum => self[node].set_value(1.0),
                    _ => (),
                }
            }
        }
        if let Some(ref mut s) = self.solver {
            match s {
                Solver::SMinInDegree(ref mut solver) => solver.reset_cache(),
                Solver::SMinOutDegree(ref mut solver) => solver.reset_cache(),
                Solver::SMaxDegree(ref mut solver) => solver.reset_cache(),
                Solver::SVSIDS(ref mut solver) => solver.reset_cache(),
                Solver::QMinInDegree(ref mut solver) => solver.reset_cache(),
                Solver::QMinOutDegree(ref mut solver) => solver.reset_cache(),
                Solver::QMaxDegree(ref mut solver) => solver.reset_cache(),
                Solver::QVSIDS(ref mut solver) => solver.reset_cache(),
            }
        }
    }

    /// Evaluates the circuits, layer by layer (starting from the input distribution, then layer 0)
    pub fn evaluate(&mut self) -> Float {
        self.reset();
        for node in (0..self.nodes.len()).map(NodeIndex) {
            self.send_value(node, self[node].get_value());
        }
        // Last node is the root since it has the higher layer
        self.nodes.last().unwrap().get_value()
    }
    
    // --- Various helper methods for the python bindings ---
    

    // --- GETTERS --- //
    
    /// Returns the probability of the circuit. If the circuit has not been evaluated,
    /// 1.0 is returned if the root is a multiplication node, and 0.0 if the root is a
    /// sum node
    pub fn get_circuit_probability(&self) -> Float {
        self.nodes.last().unwrap().get_value()
    }
    
    /// Returns the node at the given index in the input vector
    pub fn get_input_at(&self, index: usize) -> NodeIndex {
        self.inputs[index]
    }
    
    /// Returns the node at the given index in the output vector
    pub fn get_output_at(&self, index: usize) -> NodeIndex {
        self.outputs[index]
    }
    
    // Retruns, for a given distribution index and its value, the corresponding node index in the dac
    pub fn get_distribution_value_node_index(&mut self, distribution: DistributionIndex, value: usize, probability: f64) -> NodeIndex {
        if let Some(x) = self.distribution_mapping.get(&(distribution, value)) {
            *x
        }
        else {
            self.nodes.push(Node::distribution(distribution.0, value, probability));
            self.distribution_mapping.insert((distribution, value), NodeIndex(self.nodes.len()-1));
            NodeIndex(self.nodes.len()-1)
        }
    }

    // --- QUERIES --- //
    
    /// Returns the number of computational nodes in the circuit
    pub fn number_nodes(&self) -> usize {
        self.nodes.len()
    }
    
    /// Returns the number of input distributions in the circuit
    pub fn number_distributions(&self) -> usize {
        let mut cnt = 0;
        for node in (0..self.nodes.len()).map(NodeIndex) {
            if self[node].is_distribution() {
                cnt += 1;
            }
        }
        cnt
    }

    /// Returns true iff the circuit has some nodes that have been cut-of during expansion.
    /// These node contains the propagations that have been done
    pub fn has_cutoff_nodes(&self) -> bool {
        // TODO: Use a flag
        self.nodes.iter().find(|n| n.is_node_incomplete()).is_some()
    }
}

// --- ITERATOR ---

impl Dac {
    pub fn iter(&self) -> impl Iterator<Item = NodeIndex> {
        (0..self.nodes.len()).map(NodeIndex)
    }

    pub fn iter_rev(&self) -> impl Iterator<Item = NodeIndex> {
        (0..self.nodes.len()).rev().map(NodeIndex)
    }
}


// --- Indexing the circuit --- 

impl std::ops::Index<NodeIndex> for Dac {
    type Output = Node;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl std::ops::IndexMut<NodeIndex> for Dac {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        &mut self.nodes[index.0]
    }
}

// Various methods for dumping the compiled diagram, including standardized format and graphviz (inspired from https://github.com/xgillard/ddo )

// Visualization as graphviz DOT file
impl Dac {
    
    fn distribution_node_id(&self, node: NodeIndex) -> usize {
        node.0
    }
    
    fn sp_node_id(&self, node: NodeIndex) -> usize {
        node.0
    }
    
    pub fn as_graphviz(&self) -> String {
        
        let dist_node_attributes = String::from("shape=circle,style=filled");
        let prod_node_attributes = String::from("shape=circle,style=filled");
        let sum_node_attributes = String::from("shape=circle,style=filled");
        let partial_node_attributes = String::from("shape=square,style=filled");
        
        let mut out = String::new();
        out.push_str("digraph {\ntranksep = 3;\n\n");
        
        // Generating the nodes in the network 

        for node in (0..self.nodes.len()).map(NodeIndex) {
            if let TypeNode::Distribution { d, v } = self[node].get_type() {
                if self[node].has_output() {
                    let id = self.distribution_node_id(node);
                    out.push_str(&Dac::node(id, &dist_node_attributes, &format!("id{} d{} v{} ({:.4})", id, d, v, self[node].get_value())));
                }
            }
        }

        for node in (0..self.nodes.len()).map(NodeIndex) {
            let id = self.sp_node_id(node);
            if self.nodes[node.0].get_propagation().len() > 0 {
                out.push_str(&Dac::node(id, &partial_node_attributes, &format!("partial{} ({:.3})", id, self[node].get_value())));
            } else if self[node].is_product() {
                out.push_str(&Dac::node(id, &prod_node_attributes, &format!("X ({:.3})", self[node].get_value())));
            } else if self[node].is_sum() {
                out.push_str(&Dac::node(id, &sum_node_attributes, &format!("+ ({:.3})", self[node].get_value())));
            }
        }
        
        // Generating the edges
        for node in (0..self.nodes.len()).map(NodeIndex) {
            let from = self.sp_node_id(node);
            let out_start = self[node].get_output_start();
            let out_end = out_start + self[node].get_number_outputs();
            for output_i in out_start..out_end {
                let output = self.outputs[output_i];
                let to = self.sp_node_id(output);
                out.push_str(&Dac::edge(from, to, None));
            }
        }
        out.push_str("}\n");
        out
    }
    
    fn node(id: usize, attributes: &String, label: &String) -> String {
        format!("\t{id} [{attributes},label=\"{label}\"];\n")
    }
    
    fn edge(from: usize, to: usize, label: Option<String>) -> String {
        let label = if let Some(v) = label {
            v
        } else {
            String::new()
        };
        format!("\t{from} -> {to} [penwidth=1,label=\"{label}\"];\n")
    }
}

// Custom text format for the network

impl fmt::Display for Dac {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "outputs")?;
        for output in self.outputs.iter().copied() {
            write!(f, " {}", output.0)?;
        }
        writeln!(f)?;
        write!(f, "inputs")?;
        for input in self.inputs.iter().copied() {
            write!(f, " {}", input.0)?;
        }
        writeln!(f)?;
        for node in self.nodes.iter() {
            match node.get_type() {
                TypeNode::Product => write!(f, "x")?,
                TypeNode::Sum => write!(f, "+")?,
                TypeNode::Distribution {d, v} => write!(f, "d {} {}", d, v)?,
            }
            write!(f, " {} {} {} {} ", node.get_output_start(), node.get_number_outputs(), node.get_input_start(), node.get_number_inputs())?;
            writeln!(f, "{}", node.get_propagation().iter().map(|l| format!("{} {}", l.0.0, if l.1 { "t" } else { "f" })).collect::<Vec<String>>().join(" "))?;
        }
        fmt::Result::Ok(())
    }
}

impl Dac {

    pub fn from_file(filepath: &PathBuf) -> Self {
        let mut dac = Self {
            nodes: vec![],
            outputs: vec![],
            inputs: vec![],
            distribution_mapping: FxHashMap::default(),
            solver: None,
        };
        let file = File::open(filepath).unwrap();
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let l = line.unwrap();
            let split = l.split_whitespace().collect::<Vec<&str>>();
            if l.starts_with("inputs") {
                dac.inputs = split.iter().skip(1).map(|i| NodeIndex(i.parse::<usize>().unwrap())).collect();
            } else if l.starts_with("outputs") {
                dac.outputs = split.iter().skip(1).map(|i| NodeIndex(i.parse::<usize>().unwrap())).collect();
            } else if l.starts_with('x') || l.starts_with('+') {
                let values = split.iter().skip(1).take(4).map(|i| i.parse::<usize>().unwrap()).collect::<Vec<usize>>();
                let mut propagation = vec![];
                let mut i = 5;
                while i < split.len() {
                    let var = VariableIndex(split[i].parse::<usize>().unwrap());
                    let value = split[i+1] == "t";
                    propagation.push((var, value));
                    i += 2;
                }
                let mut node = if l.starts_with('x') { Node::product() } else { Node::sum() };
                node.set_output_start(values[0]);
                node.set_number_outputs(values[1]);
                node.set_input_start(values[2]);
                node.set_number_inputs(values[3]);
                dac.nodes.push(node);
            } else if l.starts_with('d') {
                let values = split.iter().skip(1).map(|i| i.parse::<usize>().unwrap()).collect::<Vec<usize>>();
                let d = values[0];
                let v = values[1];
                let mut node = Node::distribution(d, v, 0.5);
                node.set_output_start(values[2]);
                node.set_number_outputs(values[3]);
                node.set_input_start(values[4]);
                node.set_number_inputs(values[5]);
                dac.distribution_mapping.insert((DistributionIndex(d), v), NodeIndex(dac.nodes.len()-1));
            } else if l.starts_with("evaluate") {
                continue;
            }
            else if !l.is_empty() {
                panic!("Bad line format: {}", l);
            }
        }
        dac
    }
}
