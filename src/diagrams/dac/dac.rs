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

//! This module provide an implementation of an arithmetic circuit (AC) whose input are distributions.
//! As traditional AC, internal nodes are either sum or product node. The circuit is evaluated
//! bottom-up.
//! Leaves can either be inputs from a distribution, or approximate probabilities if the circuit is
//! partially compiled.
//! The circuit is constructed by a compiler, that follows the same structure of the search solver
//! but storing its trace.
//! Since the construction is done in two phases (first the circuit is constructed, then its
//! structure optimized), some information is first stored in the nodes and then transfered into
//! the vectors of the circuit.
//! Since, for now, the compilation is done using the same structure as the search, compilation
//! should only be used for learning parameters. For one shot evaluation, the search solver should
//! be used.
//!
//! The circuit is generic over the semiring on which it is evaluated.

use std::fmt;
use std::path::PathBuf;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use rustc_hash::FxHashMap;
use crate::common::*;

use crate::diagrams::*;
use crate::core::problem::{DistributionIndex, VariableIndex};
use crate::diagrams::semiring::*;
use crate::solvers::Solution;
use crate::solvers::Bounds;

use super::node::*;
use rug::Float;


/// Structure representing the arithmetic circuit.
pub struct Dac<R>
    where R: SemiRing
{
    /// Internal nodes of the circuit
    nodes: Vec<Node<R>>,
    /// Each node has a reference (two usizes) to a slice of this vector to store the index of
    /// their outputs in the circuit
    outputs: Vec<NodeIndex>,
    /// Each node has a reference (two usizes) to a slice of this vector to store the index of
    /// their inputs in the circuit
    inputs: Vec<NodeIndex>,
    /// Mapping between the (distribution, value) and the node index for distribution nodes
    distribution_mapping: FxHashMap<(DistributionIndex, usize), NodeIndex>,
    /// Root of the circuit
    root: Option<NodeIndex>,
    /// Index of the first node that is not an input
    start_computational_nodes: usize,
    /// How much seconds was needed to compile this diagram
    compile_time: u64,
}

impl<R> Dac<R>
    where R: SemiRing
{

    /// Creates a new empty DAC. An input node is created for each distribution in the problem.
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            outputs: vec![],
            inputs: vec![],
            distribution_mapping: FxHashMap::default(),
            root: None,
            start_computational_nodes: 0,
            compile_time: u64::MAX,
        }
    }

    pub fn set_compile_time(&mut self, compile_time: u64) {
        self.compile_time = compile_time;
    }

    /// Returns the number of nodes in the circuit
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Adds a prod node to the circuit. Returns its index.
    pub fn add_prod_node(&mut self) -> NodeIndex {
        let id = NodeIndex(self.nodes.len());
        self.nodes.push(Node::product());
        id
    }
    
    /// Adds a sum node to the circuit. Returns its index.
    pub fn add_sum_node(&mut self) -> NodeIndex {
        let id = NodeIndex(self.nodes.len());
        self.nodes.push(Node::sum());
        id
    }

    /// Adds a partial node, with the given value, to the circuit. Returns its index.
    pub fn add_approximate_node(&mut self, value: f64, bounds: Bounds) -> NodeIndex {
        let id = NodeIndex(self.nodes.len());
        self.nodes.push(Node::approximate(value, bounds));
        id
    }

    /// Adds `output` to the outputs of `node` and `node` to the inputs of `output`. Note that this
    /// function uses the vectors in each node. They are transferred afterward in the `outputs` vector.
    pub fn add_node_output(&mut self, node: NodeIndex, output: NodeIndex) {
        self[node].add_output(output);
        self[output].add_input(node);
    }

    /// Sets the root of the circuit
    pub fn set_root(&mut self, root: NodeIndex) {
        self.root = Some(root);
    }

    /// Returns the number of computational nodes in the circuit
    pub fn number_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the node at the given index in the input vector
    pub fn input_at(&self, index: usize) -> NodeIndex {
        self.inputs[index]
    }
    
    /// Returns the node at the given index in the output vector
    pub fn output_at(&self, index: usize) -> NodeIndex {
        self.outputs[index]
    }
    
    /// Returns, for a given distribution index and its value, the corresponding node index in the dac
    /* pub fn distribution_value_node_index(&mut self, distribution: DistributionIndex, value: usize, probability: f64) -> NodeIndex {
        if let Some(x) = self.distribution_mapping.get(&(distribution, value)) {
            *x
        } else {
            self.nodes.push(Node::distribution(distribution.0, value, probability));
            self.distribution_mapping.insert((distribution, value), NodeIndex(self.nodes.len()-1));
            NodeIndex(self.nodes.len()-1)
        }
    } */
    pub fn distribution_value_node_index(&mut self, distribution: DistributionIndex, value: usize, old_distribution: DistributionIndex, old_value: usize, probability: f64) -> NodeIndex {
        if let Some(x) = self.distribution_mapping.get(&(distribution, value)) {
            *x
        } else {
            self.nodes.push(Node::distribution(old_distribution.0, old_value, probability));
            self.distribution_mapping.insert((distribution, value), NodeIndex(self.nodes.len()-1));
            NodeIndex(self.nodes.len()-1)
        }
    }
}

// --- CIRCUIT EVALUATION ---

/// All methods in this impl block assume that the circuit's structure has been optimized and
/// layerized.
impl<R> Dac<R>
    where R: SemiRing
{
    /// Returns the probability of the circuit. If the circuit has not been evaluated,
    /// 1.0 is returned if the root is a multiplication node, and 0.0 if the root is a
    /// sum node
    pub fn circuit_probability(&self) -> &R {
        self.nodes.last().unwrap().value()
    }

    pub fn solution(&self) -> Solution {
        let p = self.circuit_probability().to_f64();
        let bounds = self.bounds();
        Solution::new(bounds.0, bounds.1, self.compile_time)
    }

    /// Returns the bounds of the circuit.
    pub fn bounds(&self) -> Bounds {
        self[NodeIndex(self.nodes.len()-1)].bounds()
    }

    /// Updates the values of the distributions to the given values
    pub fn reset_distributions(&mut self, distributions: &Vec<Vec<R>>) {
        for node in (0..self.nodes.len()).map(NodeIndex) {
            if let TypeNode::Distribution { d, v } = self[node].get_type() {
                let value = R::copy(&distributions[d][v]);
                self[node].set_value(value);
            }
            // All the distributions are at layer 0
            if self[node].layer() > 0 {
                break;
            }
        }
    }

    /// Resets the path value of each node
    pub fn zero_paths(&mut self) {
        for node in (0..self.nodes.len()-1).map(NodeIndex) {
            self[node].set_path_value(F128!(0.0));
        }
        self.nodes.last_mut().unwrap().set_path_value(F128!(1.0));
    }

    /// Evaluates the circuits, layer by layer (starting from the input distribution, then layer 0)
    pub fn evaluate(&mut self) -> &R {
        for node in (self.start_computational_nodes..self.nodes.len()).map(NodeIndex) {
            let start = self.nodes[node.0].input_start();
            let end = start + self.nodes[node.0].number_inputs();
            if self[node].is_product() {
                let value = R::mul_children((start..end).map(|idx| {
                    let child = self.inputs[idx];
                    self[child].value()
                }));
                self[node].set_value(value);
            } else {
                let value = R::sum_children((start..end).map(|idx| {
                    let child = self.inputs[idx];
                    self[child].value()
                }));
                self[node].set_value(value);
            }
        }
        // Last node is the root since it has the higher layer
        self.nodes.last().unwrap().value()
    }

}

// --- STRUCTURE OPTIMIZATION ---

impl<R> Dac<R>
    where R: SemiRing
{
    /// Optimize the structure of the circuit. This implies the following actions.
    ///     1. Every node not useful to the evaluation of the circuits are removed (e.g. a node
    ///        that has no path to the root
    ///     2. The circuit is reduce in size, by removing nodes not doing any computations (i.e.,
    ///        nodes with only one input)
    ///     3. The internal vector `nodes` is reorganized by layers.
    pub fn optimize_structure(&mut self) {
        self.remove_dead_ends();
        self.reduce();
        self.layerize();
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
        // First, we need to add the layer to each node. The nodes at layer 0 are the distribution
        // nodes or the approximate nodes.
        let mut to_process: Vec<(NodeIndex, usize)> = vec![];
        for node in (0..self.nodes.len()).map(NodeIndex) {
            if self[node].is_distribution() || self[node].is_approximate() {
                to_process.push((node, 0));
                self[node].set_layer(0);
            }
        }
        let mut number_layers = 1;
        // Then the nodes in the queue are processed. Each node give to its parent a layer of l+1
        // (l being the layer of the node). The layer is updated only if l+1 is greater than the
        // current layer of the node
        while let Some((node, layer)) = to_process.pop() {
            if number_layers < layer + 1 {
                number_layers = layer + 1;
            }
            for output_id in self[node].iter_output() {
                let output = self[node].output_at(output_id);
                if self[output].layer() < layer + 1 {
                    to_process.push((output, layer+1));
                    self[output].set_layer(layer+1);
                }
            }
        }

        // Finally, we reorganize the nodes vector by layer. It means that we need to sort its
        // content using the layers computed above.
        // To do so, we keep track, for each node, of its new and old index. This helps us update
        // the input/output vectors.
        // During this reorganization, we also delete all nodes that are not useful for the
        // evaluation of the circuit.

        let n_nodes = self.nodes.len();
        // new_indexes stores at each old index i the new index of the node.
        // The new index of a node with index i is new_indexes[i].
        let mut new_indexes = (0..n_nodes).collect::<Vec<usize>>();
        // old_indexes store for each each new index i the old index of the node.
        // This vector is used to be able to update new_indexes when nodes have been moved.
        let mut old_indexes = (0..n_nodes).collect::<Vec<usize>>();
        // Actual end of the circuits. After this reorganization, everything after end can be
        // removed
        let mut end = self.nodes.len();
        // Helps to keep track to where we need to put the nodes
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
        // TODO: This requires to do multiple pass on the
        // circuits, could probably be improved by keeping tracks of the size of each layer. Then
        // we can infer the start-end positions in the nodes vector and move every node in one pass
        // over the circuit.
        let mut start_layer = 0;
        for layer in 0..number_layers {
            if layer == 1 {
                self.start_computational_nodes = start_layer;
            }
            for i in start_layer..end{
                // If the node is part of the layer being process, then we swap it to the layer area
                let node = NodeIndex(i);
                if self[node].layer() == layer {
                    // Only move if necessary
                    self.swap(&mut new_indexes, &mut old_indexes, start_layer, i);
                    start_layer += 1;
                }
            }
        }
        
        // At this point, the nodes vector is sorted by layer. But the indexes for the outputs/inputs must be updated.
        // During this pass, the inputs/ouputs are update and moved inside the input/output vectors
        // in the DAC.
        // At this point we do not need to store them in the nodes anymore, and we can move them.
        // One of the advantage of doing so is that all the inputs/outputs are stored in the same
        // array and each node only store two usize (the start/size of the slice in which their
        // parents/children are stored).
        for node in (0..end).map(NodeIndex) {
            // Update the outputs with the new indexes
            let n = self.outputs.len();
            self[node].set_output_start(n);
            self[node].set_number_outputs(0);

            for output_id in self[node].iter_output() {
                let output = self[node].output_at(output_id);
                // If the new index of the node is after then end, then it is dropped. Otherwise,
                // we need to update it
                if new_indexes[output.0] < end {
                    let new_output = NodeIndex(new_indexes[output.0]);
                    self.outputs.push(new_output);
                    self[node].increment_number_output();
                }
            }
            self[node].clear_and_shrink_output();

            let n = self.inputs.len();
            self[node].set_input_start(n);
            self[node].set_number_inputs(0);

            for input_id in self[node].iter_input() {
                let input = self[node].input_at(input_id);
                // If the new index of the node is after then end, then it is dropped. Otherwise,
                // we need to update it
                if new_indexes[input.0] < end {
                    let new_input = NodeIndex(new_indexes[input.0]);
                    self.inputs.push(new_input);
                    self[node].increment_number_input();
                }
            }
            self[node].clear_and_shrink_input();
        }

        // Actually remove the nodes (and allocated space) from the nodes vector.
        self.nodes.truncate(end);
        self.nodes.shrink_to(end);
        self.distribution_mapping.clear();
    }
    
    /// Tags all the node not useful for the evaluation of the circuit as to be removed.
    /// This include all nodes that are not reachable from the root.
    pub fn remove_dead_ends(&mut self) {
        let mut to_process: Vec<NodeIndex> = vec![];
        to_process.push(self.root.unwrap());
        while let Some(node) = to_process.pop() {
            if !self[node].is_to_remove() {
                continue;
            }
            self[node].set_to_remove(false);
            for child_id in self[node].iter_input() {
                let child = self[node].input_at(child_id);
                if self[child].is_to_remove() {
                    to_process.push(child);
                }
            }
        }
    }

    /// A node is said to be neutral if it is not to be removed, has only one input, and is not the root of the DAC.
    /// In such case it can be bypass (it does not change its input)
    fn is_neutral(&self, node: NodeIndex) -> bool {
        !self[node].is_to_remove() && self[node].has_output() && self[node].number_inputs() == 1
    }
    
    /// Reduces the current circuit. This implies the following transformations (in that order)
    ///     1. Neutral nodes are tagged to be removed. A neutral node is a computation node (sum/prod nodes)
    ///        which i) are not the root node ii) have only one input. Such node do not modify their
    ///        input. Hence the output can be redirected to the node's output.
    ///     2. If a sum node has as input a distribution and all its value, it can be removed. In practice it only
    ///        has input from that distribution
    pub fn reduce(&mut self) {
        // First, we remove all neutral node. Since it can lead to a possible optimization of the sum node, we do that first
        for node in (0..self.nodes.len()).map(NodeIndex) {
            if self.is_neutral(node) {
                self[node].set_to_remove(true);
                let input = self[node].iter_input().next().unwrap();
                let child = self[node].input_at(input);
                // Removing the node from the output of the input node
                if let Some(idx) = self[child].iter_output().position(|x| self[child].output_at(x) == node) {
                    self[child].remove_index_from_output(idx);
                }
                for output_id in self[node].iter_output() {
                    let output = self[node].output_at(output_id);
                    // Adds the input of node to its parent input and remove node from the inputs
                    self[child].add_output(output);
                    self[output].add_input(child);
                }
                self[node].clear_and_shrink_output();
                self[node].clear_and_shrink_input();
            }
        }

        // If a sum node sums all the value of a distribution, then it always sends 1.0 to its
        // output (which must be a prod node). Hence it can be removed safely.
        // Note that the gradients of the distribution nodes will cancel each others, hence keeping
        // them would have no impact on the learning.
        for node in (0..self.nodes.len()).map(NodeIndex) {
            if self[node].is_sum() {
                let mut in_count = 0.0;
                let mut distri_child: Option<usize> = None;
                for input_id in self[node].iter_input() {
                    let input = self[node].input_at(input_id);
                    if let TypeNode::Distribution { d, v:_ } = self[input].get_type() {
                        if distri_child.is_none() {
                            distri_child = Some(d);
                            in_count += self[input].value().to_f64();
                        } else if Some(d) == distri_child {
                            in_count += self[node].value().to_f64();
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if in_count == 1.0 {
                    self[node].set_to_remove(true);
                    self[node].clear_and_shrink_input();
                }
            }
        }
    }
}

// --- ITERATOR ---

impl<R> Dac<R>
where R: SemiRing
{
    pub fn iter(&self) -> impl Iterator<Item = NodeIndex> {
        (0..self.nodes.len()).map(NodeIndex)
    }

    pub fn iter_rev(&self) -> impl Iterator<Item = NodeIndex> {
        (0..self.nodes.len()).rev().map(NodeIndex)
    }
}


// --- Indexing the circuit --- 

impl<R> std::ops::Index<NodeIndex> for Dac<R>
where R: SemiRing
{
    type Output = Node<R>;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl<R> std::ops::IndexMut<NodeIndex> for Dac<R>
where R: SemiRing
{
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        &mut self.nodes[index.0]
    }
}

// Various methods for dumping the compiled circuits, including standardized format and problemviz (inspired from https://github.com/xgillard/ddo )

// Visualization as problemviz DOT file
impl<R> Dac<R>
where R: SemiRing
{

    /// Returns a DOT representation of the circuit
    pub fn as_graphviz(&self) -> String {

        let dist_node_attributes = String::from("shape=doublecircle,style=filled");
        let prod_node_attributes = String::from("shape=square,style=filled");
        let sum_node_attributes = String::from("shape=circle,style=filled");
        let approx_node_attributes = String::from("shape=house,style=filled");

        let mut out = String::new();
        out.push_str("digraph {\ntranksep = 3;\n\n");

        // Generating the nodes in the network 
        for node in (0..self.nodes.len()).map(NodeIndex) {
            let id = node.0;
            let value = format!("{:.4}", self[node].value().to_f64());
            match self[node].get_type() {
                TypeNode::Sum => {
                    let attributes = &sum_node_attributes;
                    let label = format!("+ {}", value);
                    out.push_str(&format!("\t{id} [{attributes},label=\"{label}\"];\n"));
                },
                TypeNode::Product => {
                    let attributes = &prod_node_attributes;
                    let label = format!("x {}", value);
                    out.push_str(&format!("\t{id} [{attributes},label=\"{label}\"];\n"));
                },
                TypeNode::Distribution{d ,v } => {
                    let attributes = &dist_node_attributes;
                    let label = format!("D {} (d{} v{})", value, d, v);
                    out.push_str(&format!("\t{id} [{attributes},label=\"{label}\"];\n"));
                },
                TypeNode::Approximate => {
                    let attributes = &approx_node_attributes;
                    let label = format!("A {}", value);
                    out.push_str(&format!("\t{id} [{attributes},label=\"{label}\"];\n"));
                }
            }
        }

        // Generating the edges
        for node in (self.start_computational_nodes..self.nodes.len()).map(NodeIndex) {
            let input_start = self[node].input_start();
            let input_end = input_start + self[node].number_inputs();
            for child_id in input_start..input_end {
                let child = self.inputs[child_id];
                let from = child.0;
                let to = node.0;
                out.push_str(&format!("\t{from} -> {to} [penwidth=1];\n"));
            }
        }
        out.push_str("}\n");
        out
    }

    /* pub fn as_graphviz(&self) -> String {

        let dist_node_attributes = String::from("shape=doublecircle,style=filled");
        let prod_node_attributes = String::from("shape=square,style=filled");
        let sum_node_attributes = String::from("shape=circle,style=filled");
        let approx_node_attributes = String::from("shape=house,style=filled");

        let mut out = String::new();
        out.push_str("digraph {\ntranksep = 3;\n\n");

        // Generating the nodes in the network 
        for node in (0..self.nodes.len()).map(NodeIndex) {
            let id = node.0;
            let value = format!("{:.4}", self[node].value().to_f64());
            match self[node].get_type() {
                TypeNode::Sum => {
                    let attributes = &sum_node_attributes;
                    let label = format!("+ {}", value);
                    out.push_str(&format!("\t{id} [{attributes},label=\"{label}\"];\n"));
                },
                TypeNode::Product => {
                    let attributes = &prod_node_attributes;
                    let label = format!("x {}", value);
                    out.push_str(&format!("\t{id} [{attributes},label=\"{label}\"];\n"));
                },
                TypeNode::Distribution{d:_ ,v:_ } => {
                    let attributes = &dist_node_attributes;
                    let label = format!("D {}", value);
                    out.push_str(&format!("\t{id} [{attributes},label=\"{label}\"];\n"));
                },
                TypeNode::Approximate => {
                    let attributes = &approx_node_attributes;
                    let label = format!("A {}", value);
                    out.push_str(&format!("\t{id} [{attributes},label=\"{label}\"];\n"));
                }
            }
        }

        // Generating the edges
        for node in (0..self.nodes.len()).map(NodeIndex) {
            for child in self[node].inputs() {
                let from = child.0;
                let to = node.0;
                out.push_str(&format!("\t{from} -> {to} [penwidth=1];\n"));
            }
        }
        out.push_str("}\n");
        out
    } */

}

// Custom text format for the network

impl<R> fmt::Display for Dac<R>
where R: SemiRing
{
    /// Formats the circuit in the following formats
    ///     1. The first line starts with "output" followed by the indexes in the outputs vector,
    ///        separated by a space
    ///     2. The first line starts with "input" followed by the indexes in the inputs vector,
    ///        separated by a space
    ///     3. There is one line per node in the diagram, starting with "x" for product node, "+"
    ///        for sum nodes, "a" for approximate node and "d" for distribution node. For
    ///        approximate and distribution node, their parameters are written next to them.
    ///        Then, for all types of node, their slice (start and size) in the inputs/outputs
    ///        vector are written
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "outputs")?;
        for output in self.outputs.iter().copied().map(|node| format!("{}", node.0)) {
            write!(f, " {}", output)?;
        }
        writeln!(f)?;
        write!(f, "inputs")?;
        for input in self.inputs.iter().copied().map(|node| format!("{}", node.0)) {
            write!(f, " {}", input)?;
        }
        writeln!(f)?;
        for node in self.nodes.iter() {
            match node.get_type() {
                TypeNode::Product => write!(f, "x")?,
                TypeNode::Sum => write!(f, "+")?,
                TypeNode::Approximate => write!(f, "a {}", node.value().to_f64())?,
                TypeNode::Distribution {d, v} => write!(f, "d {} {}", d, v)?,
            }
            writeln!(f, " {} {} {} {}", node.output_start(), node.number_outputs(), node.input_start(), node.number_inputs())?;
        }
        fmt::Result::Ok(())
    }
}

impl<R> Dac<R>
where R: SemiRing
{
    /// Dump the structure of the DAC in the given file
    pub fn to_file(&self, filepath: &PathBuf) {
        let mut output = File::create(filepath).unwrap();
        write!(output, "{}", self).unwrap();
    }

    /// Reads the structure of the dac from the given file
    pub fn from_file(filepath: &PathBuf) -> Self {
        let mut dac = Self {
            nodes: vec![],
            outputs: vec![],
            inputs: vec![],
            distribution_mapping: FxHashMap::default(),
            root: None,
            start_computational_nodes: 0,
            compile_time: 0,
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
                let mut node: Node<Float> = Node::distribution(d, v, 0.5);
                node.set_output_start(values[2]);
                node.set_number_outputs(values[3]);
                node.set_input_start(values[4]);
                node.set_number_inputs(values[5]);
                dac.distribution_mapping.insert((DistributionIndex(d), v), NodeIndex(dac.nodes.len()-1));
            } else if l.starts_with('p') {
                let values = split.iter().skip(1).map(|i| i.parse::<f64>().unwrap()).collect::<Vec<f64>>();
                let value = values[0];
                let p_in = values[1];
                let p_out = values[2];
                let node = Node::approximate(value, (F128!(p_in), F128!(p_out)));
                dac.nodes.push(node);
            }
            else if !l.is_empty() {
                panic!("Bad line format: {}", l);
            }
        }
        dac
    }
}
