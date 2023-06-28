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

use crate::core::graph::{DistributionIndex, Graph};
use rug::{Assign, Float};
use crate::common::f128;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CircuitNodeIndex(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DistributionNodeIndex(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LayerIndex(usize);

/// An internal node of the circuits
struct CircuitNode {
    /// Value of the node. Initialized at 1.0 (0.0) for product (sum) nodes, after the evaluation it is equal to
    /// the product (sum) of its input values.
    value: Float,
    /// Outputs of the node. Only used during the creation. These values are moved to the DAC structure before evaluation
    outputs: Vec<CircuitNodeIndex>,
    /// Inputs of the node. Only used during the creation to minimize the size of the circuit
    inputs: Vec<CircuitNodeIndex>,
    /// Input distributions of the node
    input_distributions: Vec<(DistributionIndex, usize)>,
    /// Is the node a product node?
    is_mul: bool,
    /// Start index of the output in the DAC's output vector
    ouput_start: usize,
    /// Number of outputs of the node
    number_outputs: usize,
    /// Layer of the network. Only use to re-order the nodes in the circuit vector.
    layer: usize,
    /// Should the node be removed as post-processing ?
    to_remove: bool,
}

/// A distribution node, an input of the circuit. Each distribution node holds the distribution's parameter as well as the outputs.
/// For each output node, it also stores the value that must be sent to the output (as an index of the probability vector).
struct DistributionNode {
    /// Probabilities of the distribution
    probabilities: Vec<f64>,
    /// Outputs of the node
    outputs: Vec<(CircuitNodeIndex, usize)>,
}

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
pub struct DAC {
    /// Internal nodes of the circuit
    nodes: Vec<CircuitNode>,
    /// Input nodes of the circuit
    distribution_nodes: Vec<DistributionNode>,
    /// Outputs of the internal nodes
    outputs: Vec<CircuitNodeIndex>,
}

impl DAC {

    /// Creates a new empty DAC. An input node is created for each distribution in the graph.
    pub fn new(graph: &Graph) -> Self {
        let mut distribution_nodes: Vec<DistributionNode> = vec![];
        for distribution in graph.distributions_iter() {
            let probabilities = graph.distribution_variable_iter(distribution).map(|v| graph.get_variable_weight(v).unwrap()).collect();
            distribution_nodes.push(DistributionNode {
                probabilities,
                outputs: vec![],
             });
        }
        Self {
            nodes: vec![],
            distribution_nodes,
            outputs: vec![],
        }
    }
    
    /// Adds a prod node to the circuit
    pub fn add_prod_node(&mut self) -> CircuitNodeIndex {
        let id = CircuitNodeIndex(self.nodes.len());
        self.nodes.push(CircuitNode {
            value: f128!(1.0),
            outputs: vec![],
            inputs: vec![],
            input_distributions: vec![],
            is_mul: true,
            ouput_start: 0,
            number_outputs: 0,
            layer: 0,
            to_remove: true,
        });
        id
    }
    
    /// Adds a sum node to the circuit
    pub fn add_sum_node(&mut self) -> CircuitNodeIndex {
        let id = CircuitNodeIndex(self.nodes.len());
        self.nodes.push(CircuitNode {
            value: f128!(0.0),
            outputs: vec![],
            inputs: vec![],
            input_distributions: vec![],
            is_mul: false,
            ouput_start: 0,
            number_outputs: 0,
            layer: 0,
            to_remove: true,
        });
        id
    }
    
    /// Adds `output` to the outputs of `node` and `node` to the inputs of `output`. Note that this
    /// function uses the vectors in each node. They are transferred afterward in the `outputs` vector.
    pub fn add_spnode_output(&mut self, node: CircuitNodeIndex, output: CircuitNodeIndex) {
        self.nodes[node.0].outputs.push(output);
        self.nodes[node.0].number_outputs += 1;
        self.nodes[output.0].inputs.push(node);
    }
    
    /// Adds `output` to the outputs of the distribution's input node with the given value. Adds the (distribution, value)
    /// pair to the input of `output`
    pub fn add_distribution_output(&mut self, distribution: DistributionIndex, output: CircuitNodeIndex, value: usize) {
        let node = DistributionNodeIndex(distribution.0);
        self.distribution_nodes[node.0].outputs.push((output, value));
        self.nodes[output.0].input_distributions.push((distribution, value));
    }
    
    fn swap(&mut self, new: &mut Vec<usize>, old: &mut Vec<usize>, i: usize, j: usize) {
        self.nodes.swap(i, j);
        new[old[i]] = j;
        new[old[j]] = i;
        old.swap(i, j);
    }
    
    /// Layerizes the circuit. This function sort the internal nodes of the circuits such that an internal nodes at index i
    /// has all its input at an index j < i.
    pub fn layerize(&mut self) {
        // First, we need to add the layer to each node
        let mut to_process: Vec<(CircuitNodeIndex, usize)> = vec![];
        for i in 0..self.distribution_nodes.len() {
            for (output, _) in self.distribution_nodes[i].outputs.iter().copied() {
                // The nodes after the input are at layer 0
                to_process.push((output, 0));
            }
        }
        let mut number_layers = 1;
        while let Some((node, layer)) = to_process.pop() {
            // Only change if the layer must be increased
            if layer >= self.nodes[node.0].layer {
                if number_layers < layer + 1 {
                    number_layers = layer + 1;
                }
                self.nodes[node.0].layer = layer;
                for output in self.nodes[node.0].outputs.iter().copied() {
                    if self.nodes[output.0].layer < layer + 1 {
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
        // First we remove all nodes that are not part of a path from an  input to the root of the
        // circuit.
        while start < end {
            if self.nodes[start].to_remove {
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
                if self.nodes[i].layer == layer {
                    // Only move if necessary
                    if i != start_layer {
                        self.swap(&mut new_indexes, &mut old_indexes, start_layer, i);
                    }
                    start_layer += 1;
                }
            }
        }
        
        // At this point, the nodes vector is sorted by layer. But the indexes for the outputs must be updated.
        
        for d_node in 0..self.distribution_nodes.len() {
            // Drop all nodes that have been removed
            self.distribution_nodes[d_node].outputs.retain(|&(x,_)| new_indexes[x.0] < end);
            // Update the outputs with the new indexes
            for i in 0..self.distribution_nodes[d_node].outputs.len() {
                let (output, v) = self.distribution_nodes[d_node].outputs[i];
                let new_output = CircuitNodeIndex(new_indexes[output.0]);
                self.distribution_nodes[d_node].outputs[i] = (new_output, v);
            }
        }
        
        // Same for the internal nodes
        for node in (0..end).map(CircuitNodeIndex) {
            self.nodes[node.0].ouput_start = self.outputs.len();
            self.nodes[node.0].number_outputs = 0;
            while let Some(output) = self.nodes[node.0].outputs.pop() {
                // If the node has not been dropped, update the output. At this step, it is also moved in the
                // outputs vector of the DAC structure
                if new_indexes[output.0] < end {
                    let new_output = CircuitNodeIndex(new_indexes[output.0]);
                    self.outputs.push(new_output);
                    self.nodes[node.0].number_outputs += 1;
                }
            }
            // This remove the allocated space to the outputs/inputs vector
            self.nodes[node.0].outputs.shrink_to_fit();
            self.nodes[node.0].inputs.truncate(0);
            self.nodes[node.0].inputs.shrink_to_fit();
        }
        // Actually remove the nodes (and allocated space) from the nodes vector.
        self.nodes.truncate(end);
        self.nodes.shrink_to(end);
    }
    
    /// Tag all nodes that are not on a path from an input to the root of the DAC as to be removed
    pub fn remove_dead_ends(&mut self) {
        let mut to_process: Vec<CircuitNodeIndex> = vec![];
        for node in 0..self.distribution_nodes.len() {
            for (output, _) in self.distribution_nodes[node].outputs.iter().copied() {
                to_process.push(output)
            }
        }
        while let Some(node) = to_process.pop() {
            self.nodes[node.0].to_remove = false;
            for output in self.nodes[node.0].outputs.iter().copied() {
                if self.nodes[output.0].to_remove {
                    to_process.push(output);
                }
            }
        }
    }
    
    /// A node is said to be neutral if it is not to be removed, has only one input, and is not the root of the DAC.
    /// In such case it can be bypass (it does not change its input)
    fn is_neutral(&self, node: usize) -> bool {
        !self.nodes[node].to_remove && 
        self.nodes[node].outputs.len() != 0 &&
        self.nodes[node].inputs.len() + self.nodes[node].input_distributions.len() == 1
    }
    
    /// Reduces the current circuit. This implies the following transformations (in that order)
    ///     1. Neutral nodes are tagged to be removed. A neutral node is a computation node (sum/prod nodes)
    ///        which i) are not the root node ii) have only one input. Such node do not modify their
    ///        input. Hence the output can be redirected to the node's output.
    ///     2. If a sum node has as input a distribution and all its value, it can be removed. In practice it only
    ///        has input from that distribution
    pub fn reduce(&mut self) {
        
        // First, we remove all neutral node. Since it can lead to a possible optimization of the sum node, we do that first
        for node in 0..self.nodes.len() {
            if self.is_neutral(node) {
                self.nodes[node].to_remove = true;
                // Either it has an input from another internal node, or from a distribution node
                if self.nodes[node].inputs.len() != 0 {
                    let input = self.nodes[node].inputs[0];
                    // Removing the node from the output of the input node
                    if let Some(idx) = self.nodes[input.0].outputs.iter().position(|x| (*x).0 == node) {
                        self.nodes[input.0].outputs.remove(idx);
                    }
                    for out_id in 0..self.nodes[node].outputs.len() {
                        let output = self.nodes[node].outputs[out_id];
                        // Adds the input of node to its parent input and remove node from the inputs
                        self.nodes[input.0].outputs.push(output);
                        // Updating the input of the new output from node -> input
                        if let Some(idx) = self.nodes[output.0].inputs.iter().position(|x| (*x).0 == node) {
                            self.nodes[output.0].inputs[idx] = input;
                        }
                    }
                    self.nodes[node].inputs.clear();
                    self.nodes[node].outputs.clear();
                } else {
                    let (input, value) = self.nodes[node].input_distributions[0];
                    // Removing the node from the output of the distribution
                    if let Some(idx) = self.distribution_nodes[input.0].outputs.iter().position(|x| *x == (CircuitNodeIndex(node), value)) {
                        self.distribution_nodes[input.0].outputs.remove(idx);
                    }
                    for out_id in 0..self.nodes[node].outputs.len() {
                        // Adding the new output to the distribution's outputs
                        let output = self.nodes[node].outputs[out_id];
                        self.distribution_nodes[input.0].outputs.push((output, value));
                        // Adding the distribution to the inputs of the new output
                        self.nodes[output.0].input_distributions.push((input, value));
                        // Removing the node from the input of the output 
                        if let Some(idx) = self.nodes[output.0].inputs.iter().position(|x| (*x).0 == node) {
                            self.nodes[output.0].inputs.remove(idx);
                        }
                    }
                    self.nodes[node].input_distributions.clear();
                    self.nodes[node].outputs.clear();
                }
            }
        }
        
        // Then we simplify the sum nodes
        for node in 0..self.nodes.len() {
            // Only consider sum node that have a distribution as input
            if !self.nodes[node].is_mul && !self.nodes[node].input_distributions.is_empty() {
                // Count the number of value of the first distribution, minus 1 since we assume the first element has been processed
                let mut seen_values = self.distribution_nodes[self.nodes[node].input_distributions[0].0.0].probabilities.len() - 1;
                let mut to_remove = true;
                for i in 1..self.nodes[node].input_distributions.len() {
                    // If the distribution is different from the previous, multiple distributions so we must keep it
                    if self.nodes[node].input_distributions[i].0 != self.nodes[node].input_distributions[i-1].0 {
                        to_remove = false;
                        break;
                    }
                    seen_values += 1;
                }
                if to_remove && seen_values == 0 {
                    self.nodes[node].to_remove = true;
                }
            }
        }
    }
    
    // --- Evaluation ---- //
    
    // Resets the value of each internal node
    fn reset_nodes(&mut self) {
        for node in self.nodes.iter_mut() {
            if node.is_mul {
                node.value.assign(1.0);
            } else {
                node.value.assign(0.0);
            }
        }
    }
    
    /// Evaluates the circuits, layer by layer (starting from the input distribution, then layer 0)
    pub fn evaluate(&mut self) -> Float {
        self.reset_nodes();
        for d_node in (0..self.distribution_nodes.len()).map(DistributionNodeIndex) {
            for (output, value) in self.distribution_nodes[d_node.0].outputs.iter().copied() {
                if self.nodes[output.0].is_mul {
                    self.nodes[output.0].value *= self.distribution_nodes[d_node.0].probabilities[value];
                } else {
                    self.nodes[output.0].value += self.distribution_nodes[d_node.0].probabilities[value];                    
                }
            }
        }
        for node in (0..self.nodes.len()).map(CircuitNodeIndex) {
            let start = self.nodes[node.0].ouput_start;
            let end = start + self.nodes[node.0].number_outputs;
            let value = self.nodes[node.0].value.clone();
            for i in start..end {
                let output = self.outputs[i];
                if self.nodes[output.0].is_mul {
                    self.nodes[output.0].value *= &value;
                } else {
                    self.nodes[output.0].value += &value;                    
                }
            }
        }
        // Last node is the root since it has the higher layer
        self.nodes.last().unwrap().value.clone()
    }
}

// Various methods for dumping the compiled diagram, including standardized format and graphviz (inspired from https://github.com/xgillard/ddo )

// Visualization as graphviz DOT file
impl DAC {
    
    fn distribution_node_id(&self, node: DistributionNodeIndex) -> usize {
        node.0
    }
    
    fn sp_node_id(&self, node: CircuitNodeIndex) -> usize {
        self.distribution_nodes.len() + node.0
    }
    
    pub fn as_graphviz(&self) -> String {
        
        let dist_node_attributes = String::from("shape=circle,style=filled");
        let prod_node_attributes = String::from("shape=circle,style=filled");
        let prod_node_label = String::from("X");
        let sum_node_attributes = String::from("shape=circle,style=filled");
        let sum_node_label = String::from("+");
        
        let mut out = String::new();
        out.push_str("digraph {\ntranksep = 3;\n\n");
        
        // Generating the nodes in the network 

        for node in (0..self.distribution_nodes.len()).map(DistributionNodeIndex) {
            let id = self.distribution_node_id(node);
            out.push_str(&DAC::node(id, &dist_node_attributes, &format!("d{}", id)));
        }

        for node in (0..self.nodes.len()).map(CircuitNodeIndex) {
            let id = self.sp_node_id(node);
            if self.nodes[node.0].is_mul {
                out.push_str(&DAC::node(id, &prod_node_attributes, &prod_node_label));
            } else {
                out.push_str(&DAC::node(id, &sum_node_attributes, &sum_node_label));
            }
        }
        
        // Generating the edges
        for node in (0..self.distribution_nodes.len()).map(DistributionNodeIndex) {
            let from = self.distribution_node_id(node);
            for (output, value) in self.distribution_nodes[node.0].outputs.iter().copied() {
                let to = self.sp_node_id(output);
                let f_value = format!("{:.5}",self.distribution_nodes[node.0].probabilities[value]);
                out.push_str(&DAC::edge(from, to, Some(f_value)));
            }
        }
        
        for node in (0..self.nodes.len()).map(CircuitNodeIndex) {
            let from = self.sp_node_id(node);
            let start = self.nodes[node.0].ouput_start;
            let end = start + self.nodes[node.0].number_outputs;
            for output in self.outputs[start..end].iter().copied() {
                let to = self.sp_node_id(output);
                out.push_str(&DAC::edge(from, to, None));
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
            format!("{}", v)
        } else {
            format!("")
        };
        format!("\t{from} -> {to} [penwidth=1,label=\"{label}\"];\n")
    }
}

// Custom text format for the network

impl fmt::Display for DAC {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "dspn {} {}", self.distribution_nodes.len(), self.nodes.len())?;
        for node in self.distribution_nodes.iter() {
            write!(f, "d {}", node.probabilities.len())?;
            for p in node.probabilities.iter() {
                write!(f, " {:.8}", p)?;
            }
            for (output, value) in node.outputs.iter().copied() {
                write!(f, " {} {}", output.0, value)?;
            }
            writeln!(f, "")?;
        }
        for node in self.nodes.iter() {
            if node.is_mul {
                write!(f, "x")?;       
            } else {
                write!(f, "+")?;
            }
            for output_id in node.ouput_start..(node.ouput_start+node.number_outputs) {
                write!(f, " {}", self.outputs[output_id].0)?;
            }
            writeln!(f, "")?;
        }
        fmt::Result::Ok(())
    }
}

impl DAC {

    pub fn from_file(filepath: &PathBuf) -> Self {
        let mut spn = Self {
            nodes: vec![],
            distribution_nodes: vec![],
            outputs: vec![],
        };
        let file = File::open(filepath).unwrap();
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let l = line.unwrap();
            let split = l.split_whitespace().collect::<Vec<&str>>();
            if l.starts_with("dspn") {
                // Do things ?
            } else if l.starts_with("d") {
                let dom_size = split[1].parse::<usize>().unwrap();
                let probabilities = split[2..(2+dom_size)].iter().map(|s| s.parse::<f64>().unwrap()).collect::<Vec<f64>>();
                let mut outputs: Vec<(CircuitNodeIndex, usize)> = vec![];
                for i in ((2+dom_size)..split.len()).step_by(2) {
                    let output_node = CircuitNodeIndex(split[i].parse::<usize>().unwrap());
                    let value = split[i+1].parse::<usize>().unwrap();
                    outputs.push((output_node, value));
                }
                spn.distribution_nodes.push(DistributionNode {
                    probabilities,
                    outputs,
                });
            } else if l.starts_with("x") || l.starts_with("+") {
                let start = spn.outputs.len();
                for i in 1..split.len() {
                    spn.outputs.push(CircuitNodeIndex(split[i].parse::<usize>().unwrap()));
                }
                let end = spn.outputs.len();
                spn.nodes.push(CircuitNode {
                    value: if l.starts_with("x") { f128!(1.0) } else { f128!(0.0) },
                    outputs: vec![],
                    inputs: vec![],
                    input_distributions: vec![],
                    is_mul: l.starts_with("x"),
                    ouput_start: start,
                    number_outputs: end - start,
                    layer: 0,
                    to_remove: false,
                })
            } else if !l.is_empty() {
                panic!("Bad line format: {}", l);
            }
        }
        spn
    }
}