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
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

//use crate::core::graph::{DistributionIndex, Graph};
use rug::{Assign, Float};
use crate::{common::f128, core::graph::DistributionIndex};

//#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
//pub struct CircuitNodeIndex(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex(pub usize);

//#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
//pub struct DistributionNodeIndex(pub usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LayerIndex(usize);


#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeNode {
    Product,
    Sum,
    Distribution {d: usize, v: usize},
}
impl TypeNode {
    pub fn get_value(&self) -> &usize {
        match self {
            TypeNode::Product => panic!("Product typenode has no value"),
            TypeNode::Sum => panic!("Sum typenode has no value"),
            TypeNode::Distribution{d:_,v} => &v
        }
    }
}

/// A node structur that represents both internal and distribution nodes.
pub struct Node {
    /// Value of the node. Initialized at 1.0 for product, at 0.0 for sum and at a specific value for distribution nodes.
    /// For product and sum nodes, after the evaluation it is equal to the product (sum) of its input values.
    value: Float,
    /// Outputs of the node. Only used during the creation. These values are moved to the DAC structure before evaluation
    outputs: Vec<NodeIndex>,
    /// Inputs of the node. Only used during the creation to minimize the size of the circuit
    inputs: FxHashSet<NodeIndex>,
    /// What is the type of the node?
    typenode: TypeNode,
    /// Start index of the output in the DAC's output vector
    output_start: usize,
    /// Number of outputs of the node
    number_outputs: usize,
    /// Same as the output, but for the inputs
    input_start: usize,
    /// Same as the output, but for the inputs
    number_inputs: usize,
    /// Layer of the network. Only use to re-order the nodes in the circuit vector.
    layer: usize,
    /// Should the node be removed as post-processing ?
    to_remove: bool,
    // Gradient computation, the value of the path from the root
    path_value: Float,
}
impl Node{
    // --- Getters --- //
    pub fn get_value(&self) -> Float{
        self.value.clone()
    }
    pub fn get_path_value(&self) -> Float{
        self.path_value.clone()
    }
    pub fn get_type(&self) -> TypeNode{
        self.typenode
    }
    pub fn get_output_start(&self) -> usize{
        self.output_start
    }
    pub fn get_number_outputs(&self) -> usize{
        self.number_outputs
    }
    pub fn get_input_start(&self) -> usize{
        self.input_start
    }
    pub fn get_number_inputs(&self) -> usize{
        self.number_inputs
    }
    pub fn get_layer(&self) -> usize{
        self.layer
    }
    pub fn get_to_remove(&self) -> bool{
        self.to_remove
    }
    // --- Setters --- /
    pub fn set_value(&mut self, value: f64){
        self.value.assign(value);
    }
    pub fn set_path_value(&mut self, value: Float){
        self.path_value = value;
    }
    pub fn set_type(&mut self, typenode: TypeNode){
        self.typenode = typenode;
    }
    pub fn set_output_start(&mut self, output_start: usize){
        self.output_start = output_start;
    }
    pub fn set_number_outputs(&mut self, number_outputs: usize){
        self.number_outputs = number_outputs;
    }
    pub fn set_input_start(&mut self, input_start: usize){
        self.input_start = input_start;
    }
    pub fn set_number_inputs(&mut self, number_inputs: usize){
        self.number_inputs = number_inputs;
    }
    pub fn set_layer(&mut self, layer: usize){
        self.layer = layer;
    }
    pub fn set_to_remove(&mut self, to_remove: bool){
        self.to_remove = to_remove;
    }

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
pub struct Dac {
    /// Internal nodes of the circuit
    pub nodes: Vec<Node>,
    /// Outputs of the internal nodes
    outputs: Vec<NodeIndex>,
    /// Inputs of the internal nodes
    inputs: Vec<NodeIndex>,
    // Mapping between the (distribution, value) and the node index for distribution nodes
    distribution_mapping: FxHashMap<(DistributionIndex, usize), NodeIndex>,
}

impl Dac {

    /// Creates a new empty DAC. An input node is created for each distribution in the graph.
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            outputs: vec![],
            inputs: vec![],
            distribution_mapping: FxHashMap::default(),
        }
    }
    
    /// Adds a prod node to the circuit
    pub fn add_prod_node(&mut self) -> NodeIndex {
        let id = NodeIndex(self.nodes.len());
        self.nodes.push(Node {
            value: f128!(1.0),
            outputs: vec![],
            inputs: FxHashSet::default(),
            typenode: TypeNode::Product,
            output_start: 0,
            number_outputs: 0,
            input_start: 0,
            number_inputs: 0,
            layer: 0,
            to_remove: true,
            path_value: f128!(1.0),
        });
        id
    }
    
    /// Adds a sum node to the circuit
    pub fn add_sum_node(&mut self) -> NodeIndex {
        let id = NodeIndex(self.nodes.len());
        self.nodes.push(Node {
            value: f128!(0.0),
            outputs: vec![],
            inputs: FxHashSet::default(),
            typenode: TypeNode::Sum,
            output_start: 0,
            number_outputs: 0,
            input_start: 0,
            number_inputs: 0,
            layer: 0,
            to_remove: true,
            path_value: f128!(1.0),
        });
        id
    }
    
    /// Adds `output` to the outputs of `node` and `node` to the inputs of `output`. Note that this
    /// function uses the vectors in each node. They are transferred afterward in the `outputs` vector.
    pub fn add_node_output(&mut self, node: NodeIndex, output: NodeIndex) {
        self.nodes[node.0].outputs.push(output);
        self.nodes[node.0].number_outputs += 1;
        self.nodes[output.0].inputs.insert(node);
        //self.nodes[output.0].number_inputs += 1;
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
        for i in 0..self.nodes.len() {
            if let TypeNode::Distribution {..} = self.nodes[i].typenode {
                // The distribution nodes are at layer 0
                to_process.push((NodeIndex(i), 0));
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
        // First we remove all nodes that are not part of a path from an input to the root of the
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
        
        // At this point, the nodes vector is sorted by layer. But the indexes for the outputs/inputs must be updated.
        for node in 0..self.nodes.len() {
            // Drop all nodes that have been removed
            if let TypeNode::Distribution {..} = self.nodes[node].get_type(){self.nodes[node].outputs.retain(|&x| new_indexes[x.0] < end);}
            // Update the outputs with the new indexes
            self.nodes[node].output_start = self.outputs.len();
            self.nodes[node].number_outputs = 0;
            while let Some(output) = self.nodes[node].outputs.pop() {
                // If the node has not been dropped, update the output. At this step, it is also moved in the
                // outputs vector of the DAC structure
                if new_indexes[output.0] < end {
                    let new_output = NodeIndex(new_indexes[output.0]);
                    self.outputs.push(new_output);
                    self.nodes[node].number_outputs += 1;
                }
            }
            self.nodes[node].outputs.shrink_to_fit();
            let input_start = self.nodes[node].input_start;
            let number_input = self.nodes[node].number_inputs;
            for input_index in input_start..(input_start+number_input) {
                let old = self.inputs[input_index];
                self.inputs[input_index] = NodeIndex(new_indexes[old.0]);
            }
        }
        // Actually remove the nodes (and allocated space) from the nodes vector.
        self.nodes.truncate(end);
        self.nodes.shrink_to(end);
    }
    
    /// Tag all nodes that are not on a path from an input to the root of the DAC as to be removed
    pub fn remove_dead_ends(&mut self) {
        let mut to_process: Vec<NodeIndex> = vec![];
        for node in 0..self.nodes.len() {
            if matches!(self.nodes[node].typenode, TypeNode::Distribution{..}){
                for output_i in 0..self.nodes[node].outputs.len() {
                    let output = self.nodes[node].outputs[output_i];
                    self.nodes[node].to_remove = false;
                    to_process.push(output)
                }
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
        !self.nodes[node].outputs.is_empty() &&
        self.nodes[node].inputs.len() == 1
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
            for node in 0..self.nodes.len() {
                if self.is_neutral(node) {
                    changed = true;
                    self.nodes[node].to_remove = true;
                    // The removal of the node is the same, no matter the node type of the input
                    let input = *self.nodes[node].inputs.iter().next().unwrap();
                    // Removing the node from the output of the input node
                    if let Some(idx) = self.nodes[input.0].outputs.iter().position(|x| x.0 == node) {
                        self.nodes[input.0].outputs.remove(idx);
                    }
                    for out_id in 0..self.nodes[node].outputs.len() {
                        let output = self.nodes[node].outputs[out_id];
                        // Adds the input of node to its parent input and remove node from the inputs
                        self.nodes[input.0].outputs.push(output);
                        // Updating the input of the new output from node -> input
                        self.nodes[output.0].inputs.remove(&NodeIndex(node));
                        self.nodes[output.0].inputs.insert(input);
                    }
                    self.nodes[node].inputs.clear();
                    self.nodes[node].outputs.clear();
                
                }
            }
                
            // If a distribution node send all its value to a sum node, remove the node from the output
            for node in 0..self.nodes.len() {
                if let TypeNode::Sum = self.nodes[node].typenode {
                    let mut in_count = 0.0;
                    let mut distri_child: Option<usize> = None;
                    for input in self.nodes[node].inputs.iter().copied() {
                        if let TypeNode::Distribution { d, v:_ } = self.nodes[input.0].typenode {
                            if distri_child.is_none() {
                                distri_child = Some(d);
                                in_count += self.nodes[input.0].value.to_f64();
                            }
                            else if Some(d) == distri_child {
                                in_count += self.nodes[input.0].value.to_f64();
                            }
                            else {
                                break;
                            }
                        }
                        else {
                            break;
                        }
                    }
                    if in_count == 1.0 {
                        changed = true;
                        for input in self.nodes[node].inputs.iter().copied().collect::<Vec<NodeIndex>>() {
                            debug_assert!(matches!(self.nodes[input.0].typenode, TypeNode::Distribution{..}));
                            let index_to_remove: usize = self.nodes[input.0].outputs.iter().position(|x| x.0 == node).unwrap();
                            self.nodes[input.0].outputs.swap_remove(index_to_remove);
                        }
                        self.nodes[node].to_remove = true;
                        self.nodes[node].inputs.clear();
                        for o in 0..self.nodes[node].outputs.len() {
                            let o_output = self.nodes[node].outputs[o].0;
                            self.nodes[o_output].inputs.remove(&NodeIndex(node));
                        }
                    }
                }
            }
        }
        // Move the inputs into the input vector
        for node in 0..self.nodes.len() {
            self.nodes[node].input_start = self.inputs.len();
            self.nodes[node].number_inputs = self.nodes[node].inputs.len();
            for input in self.nodes[node].inputs.iter().copied() {
                self.inputs.push(input);
            }
            self.nodes[node].inputs.clear();
            self.nodes[node].inputs.shrink_to_fit();
        }
    }
    
    // --- Evaluation ---- //

    fn send_value(&mut self, node: NodeIndex, value: Float) {
        let start = self.nodes[node.0].get_output_start();
        let end = start + self.nodes[node.0].get_number_outputs();
        for i in start..end {
            let output = self.outputs[i];
            if matches!(self.nodes[output.0].typenode, TypeNode::Product){
                self.nodes[output.0].value *= &value;
            } else if matches!(self.nodes[output.0].typenode, TypeNode::Sum) {
                self.nodes[output.0].value += &value;                    
            }
        }
    }

    fn reset(&mut self) {
        for node in (0..self.nodes.len()).map(NodeIndex) {
            match self.nodes[node.0].typenode {
                TypeNode::Product => self.nodes[node.0].value.assign(1.0),
                TypeNode::Sum => self.nodes[node.0].value.assign(0.0),
                TypeNode::Distribution {..} => {},
            }
        }
    }

    /// Evaluates the circuits, layer by layer (starting from the input distribution, then layer 0)
    pub fn evaluate(&mut self) -> Float {
        self.reset();
        for node in (0..self.nodes.len()).map(NodeIndex) {
            self.send_value(node, self.nodes[node.0].value.clone())
        }
        // Last node is the root since it has the higher layer
        self.nodes.last().unwrap().value.clone()
    }
    
    // --- Various helper methods for the python bindings ---
    

    // --- GETTERS --- //
    
    /// Returns the index of the first output of the node, in the output vector
    pub fn get_circuit_node_out_start(&self, node: NodeIndex) -> usize {
        self.nodes[node.0].output_start    
    }

    /// Returns the number of output of the node
    pub fn get_circuit_node_number_output(&self, node: NodeIndex) -> usize {
        self.nodes[node.0].number_outputs    
    }
    
    /// Returns the index of the first input of the node, in the input vector
    pub fn get_circuit_node_in_start(&self, node: NodeIndex) -> usize {
        self.nodes[node.0].input_start
    }

    /// Returns the number of input of the node
    pub fn get_circuit_node_number_input(&self, node: NodeIndex) -> usize {
        self.nodes[node.0].number_inputs
    }
    
    /// Returns the probability of the circuit. If the node has not been evaluated,
    /// 1.0 is returned if the root is a multiplication node, and 0.0 if the root is a
    /// sum node
    pub fn get_circuit_node_probability(&self, node: NodeIndex) -> &Float {
        &self.nodes[node.0].value
    }
    
    /// Returns the probability of the circuit. If the circuit has not been evaluated,
    /// 1.0 is returned if the root is a multiplication node, and 0.0 if the root is a
    /// sum node
    pub fn get_circuit_probability(&self) -> &Float {
        &self.nodes[self.nodes.len()-1].value
    }
    
    /// Returns the node at the given index in the input vector
    pub fn get_input_at(&self, index: usize) -> NodeIndex {
        self.inputs[index]
    }
    
    /// Returns the node at the given index in the output vector
    pub fn get_output_at(&self, index: usize) -> NodeIndex {
        self.outputs[index]
    }
    
    /// Returns the number of input edges from the distributions of the given node
    pub fn get_circuit_node_number_distribution_input(&self, node: NodeIndex) -> usize {
        let mut cnt = 0;
        let num_inputs = self.nodes[node.0].number_inputs;
        let input_start = self.nodes[node.0].input_start;
        for i in 0..num_inputs {
            if matches!(self.nodes[self.inputs[input_start+i].0].typenode, TypeNode::Distribution{..}) {
                cnt += 1;
            }
        }
        cnt
    }
    
    // Retruns, for a given distribution index and its value, the corresponding node index in the dac
    pub fn get_distribution_value_node_index(&mut self, distribution: DistributionIndex, value: usize, probability: f64) -> NodeIndex {
        if let Some(x) = self.distribution_mapping.get(&(distribution, value)) {
            *x
        }
        else {
            self.nodes.push(Node {
                value: f128!(probability),
                outputs: vec![],
                inputs: FxHashSet::default(),
                typenode: TypeNode::Distribution {d: distribution.0, v: value},
                output_start: 0,
                number_outputs: 0,
                input_start: 0,
                number_inputs: 0,
                layer: 0,
                to_remove: true,
                path_value: f128!(1.0),
            });
            self.distribution_mapping.insert((distribution, value), NodeIndex(self.nodes.len()-1));
            NodeIndex(self.nodes.len()-1)
        }
    }

    /// Returns, for a given node and an index in its distributions input vector, the distribution index of the input and the value
    /// index send from the probability
    /* pub fn get_circuit_node_input_distribution_at(&self, node: NodeIndex, index: usize) -> (NodeIndex, usize) {
        self.nodes[self.inputs[node.0].input_distributions[index]]
    } */
    
    /// Returns the number of value in the given distribution
    /* pub fn get_distribution_domain_size(&self, distribution: NodeIndex) -> usize {
        self.distribution_nodes[distribution.0].unsoftmaxed_probabilities.len()
    } */
    
    /// Returns the probability at the given index of the given distribution
    /* pub fn get_distribution_probability_at(&self, distribution: DistributionNodeIndex, index: usize) -> f64 {
        let softmaxed = softmax(&self.distribution_nodes[distribution.0].unsoftmaxed_probabilities);
        softmaxed[index]
    } */
    
    /// Returns the pair (circuit_node, index) for the output of the distribution at the given index
    /* pub fn get_distribution_output_at(&self, distribution: DistributionNodeIndex, index: usize) -> (CircuitNodeIndex, usize) {
        self.distribution_nodes[distribution.0].outputs[index]
    } */
    
    /// Returns the number of output of a distribution node
    /* pub fn get_distribution_number_output(&self, distribution: NodeIndex) -> usize {
        self.nodes[distribution.0].outputs.len()
    } */
    
    // --- SETTERS --- //
    
    /// Set the unsoftmaxed probability of the distribution, at the given index, to the given value
    /* pub fn set_distribution_probability_at(&mut self, distribution: DistributionNodeIndex, index: usize, value: f64) {
        self.distribution_nodes[distribution.0].unsoftmaxed_probabilities[index] = value;
    } */


    // --- QUERIES --- //

    /// Returns true if the given node is a multiplication node
    pub fn is_circuit_node_mul(&self, node: NodeIndex) -> bool {
        if matches!(self.nodes[node.0].typenode, TypeNode::Product) {
            return true
        }
        false
    }
    
    /// Returns the number of computational nodes in the circuit
    pub fn number_nodes(&self) -> usize {
        self.nodes.len()
    }
    
    /// Returns the number of input distributions in the circuit
    pub fn number_distributions(&self) -> usize {
        let mut cnt = 0;
        for node in 0..self.nodes.len() {
            if matches!(self.nodes[node].typenode, TypeNode::Distribution{..}) {
                cnt += 1;
            }
        }
        cnt
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
        
        let mut out = String::new();
        out.push_str("digraph {\ntranksep = 3;\n\n");
        
        // Generating the nodes in the network 

        for node in (0..self.nodes.len()).map(NodeIndex) {
            if matches!(self.nodes[node.0].typenode, TypeNode::Distribution {..}) && !self.nodes[node.0].outputs.is_empty() {
                let id = self.distribution_node_id(node);
                out.push_str(&Dac::node(id, &dist_node_attributes, &format!("d{}", id)));
            }
        }

        for node in (0..self.nodes.len()).map(NodeIndex) {
            let id = self.sp_node_id(node);
            if matches!(self.nodes[node.0].typenode, TypeNode::Product) {
                out.push_str(&Dac::node(id, &prod_node_attributes, &format!("X ({:.3})", self.nodes[node.0].value)));
            } else if matches!(self.nodes[node.0].typenode, TypeNode::Sum) {
                out.push_str(&Dac::node(id, &sum_node_attributes, &format!("+ ({:.3})", self.nodes[node.0].value)));
            }
        }
        
        // Generating the edges
        for node in (0..self.nodes.len()).map(NodeIndex) {
            let from = self.distribution_node_id(node);
            for output in self.nodes[node.0].outputs.iter().filter(|n| matches!(self.nodes[n.0].typenode, TypeNode::Distribution {..})).copied() {
                let to = self.sp_node_id(output);
                let f_value = format!("({}, {:.3})", self.nodes[node.0].typenode.get_value(), self.nodes[output.0].value);
                out.push_str(&Dac::edge(from, to, Some(f_value)));
            }
        }
        
        for node in (0..self.nodes.len()).map(NodeIndex) {
            let from = self.sp_node_id(node);
                for output in self.nodes[node.0].outputs.iter().copied() {
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
            match node.typenode {
                TypeNode::Product => write!(f, "x")?,
                TypeNode::Sum => write!(f, "+")?,
                TypeNode::Distribution {d, v} => write!(f, "d {} {}", d, v)?,
            }
            writeln!(f, " {} {} {} {}", node.output_start, node.number_outputs, node.input_start, node.number_inputs)?;
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
                let values = split.iter().skip(1).map(|i| i.parse::<usize>().unwrap()).collect::<Vec<usize>>();
                dac.nodes.push(Node {
                    value: if l.starts_with('x') { f128!(1.0) } else { f128!(0.0) },
                    outputs: vec![],
                    inputs: FxHashSet::default(),
                    output_start: values[0],
                    number_outputs: values[1],
                    input_start: values[2],
                    number_inputs: values[3],
                    layer: 0,
                    to_remove: false,
                    path_value: f128!(1.0),
                    typenode: if l.starts_with('x') { TypeNode::Product } else { TypeNode::Sum },
                })
            } else if l.starts_with('d') {
                let values = split.iter().skip(1).map(|i| i.parse::<usize>().unwrap()).collect::<Vec<usize>>();
                let d = values[0];
                let v = values[1];
                dac.nodes.push(Node {
                    value: f128!(0.5),
                    outputs: vec![],
                    inputs: FxHashSet::default(),
                    output_start: values[2],
                    number_outputs: values[3],
                    input_start: values[4],
                    number_inputs: values[5],
                    layer: 0,
                    to_remove: false,
                    path_value: f128!(1.0),
                    typenode: TypeNode::Distribution {d, v},
                });
                dac.distribution_mapping.insert((DistributionIndex(d), v), NodeIndex(dac.nodes.len()-1));
            } else if !l.is_empty() {
                panic!("Bad line format: {}", l);
            }
        }
        dac
    }
}