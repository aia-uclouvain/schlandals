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
use rustc_hash::FxHashSet;

use crate::core::graph::{DistributionIndex, Graph};
use rug::{Assign, Float};
use crate::common::f128;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CircuitNodeIndex(pub usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DistributionNodeIndex(pub usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LayerIndex(usize);

/// An internal node of the circuits
pub struct CircuitNode {
    /// Value of the node. Initialized at 1.0 (0.0) for product (sum) nodes, after the evaluation it is equal to
    /// the product (sum) of its input values.
    value: Float,
    /// Outputs of the node. Only used during the creation. These values are moved to the DAC structure before evaluation
    outputs: Vec<CircuitNodeIndex>,
    /// Inputs of the node. Only used during the creation to minimize the size of the circuit
    inputs: FxHashSet<CircuitNodeIndex>,
    /// Input distributions of the node
    input_distributions: Vec<(DistributionIndex, usize)>,
    /// Is the node a product node?
    is_mul: bool,
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

/// A distribution node, an input of the circuit. Each distribution node holds the distribution's parameter as well as the outputs.
/// For each output node, it also stores the value that must be sent to the output (as an index of the probability vector).
struct DistributionNode {
    /// Probabilities of the distribution, not softmaxed
    unsoftmaxed_probabilities: Vec<f64>,
    /// Outputs of the node
    outputs: Vec<(CircuitNodeIndex, usize)>,
    // Gradient computation, the current value of the gradient of each possible value of the distribution
    grad_value: Vec<Float>,
}

/// Calculates the softmax (the normalized exponential) function, which is a generalization of the
/// logistic function to multiple dimensions.
///
/// Takes in a vector of real numbers and normalizes it to a probability distribution such that
/// each of the components are in the interval (0, 1) and the components add up to 1. Larger input
/// components correspond to larger probabilities.
/// From https://docs.rs/compute/latest/src/compute/functions/statistical.rs.html#43-46
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let sum_exp: f64 = x.iter().map(|i| i.exp()).sum();
    x.iter().map(|i| i.exp() / sum_exp).collect()
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
    nodes: Vec<CircuitNode>,
    /// Input nodes of the circuit
    distribution_nodes: Vec<DistributionNode>,
    /// Outputs of the internal nodes
    outputs: Vec<CircuitNodeIndex>,
    /// Inputs of the internal nodes
    inputs: Vec<CircuitNodeIndex>,
}

impl Dac {

    /// Creates a new empty DAC. An input node is created for each distribution in the graph.
    pub fn new(graph: &Graph) -> Self {
        let mut distribution_nodes: Vec<DistributionNode> = vec![];
        for distribution in graph.distributions_iter() {
            let mut probabilities: Vec<f64> = graph[distribution].iter_variables().map(|v| graph[v].weight().unwrap()).collect();
            for el in &mut probabilities {
                *el = el.log10();
            }
            let prob_len = probabilities.len();
            distribution_nodes.push(DistributionNode {
                unsoftmaxed_probabilities: probabilities,
                outputs: vec![],
                grad_value: vec![f128!(0.0); prob_len],
            });
        }
        Self {
            nodes: vec![],
            distribution_nodes,
            outputs: vec![],
            inputs: vec![],
        }
    }
    
    /// Adds a prod node to the circuit
    pub fn add_prod_node(&mut self) -> CircuitNodeIndex {
        let id = CircuitNodeIndex(self.nodes.len());
        self.nodes.push(CircuitNode {
            value: f128!(1.0),
            outputs: vec![],
            inputs: FxHashSet::default(),
            input_distributions: vec![],
            is_mul: true,
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
    pub fn add_sum_node(&mut self) -> CircuitNodeIndex {
        let id = CircuitNodeIndex(self.nodes.len());
        self.nodes.push(CircuitNode {
            value: f128!(0.0),
            outputs: vec![],
            inputs: FxHashSet::default(),
            input_distributions: vec![],
            is_mul: false,
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
    pub fn add_circuit_node_output(&mut self, node: CircuitNodeIndex, output: CircuitNodeIndex) {
        self.nodes[node.0].outputs.push(output);
        self.nodes[node.0].number_outputs += 1;
        self.nodes[output.0].inputs.insert(node);
    }
    
    /// Adds `output` to the outputs of the distribution's input node with the given value. Adds the (distribution, value)
    /// pair to the input of `output`
    pub fn add_distribution_output(&mut self, distribution: DistributionIndex, output: CircuitNodeIndex, value: usize) {
        let node = DistributionNodeIndex(distribution.0);
        self.distribution_nodes[node.0].outputs.push((output, value));
        self.nodes[output.0].input_distributions.push((distribution, value));
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
            self.nodes[node.0].output_start = self.outputs.len();
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
            self.nodes[node.0].outputs.shrink_to_fit();
            let input_start = self.nodes[node.0].input_start;
            let number_input = self.nodes[node.0].number_inputs;
            for input_index in input_start..(input_start+number_input) {
                let old = self.inputs[input_index];
                self.inputs[input_index] = CircuitNodeIndex(new_indexes[old.0]);
            }
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
        !self.nodes[node].outputs.is_empty() &&
        self.nodes[node].inputs.len() + self.nodes[node].input_distributions.len() == 1
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
                    // Either it has an input from another internal node, or from a distribution node
                    if !self.nodes[node].inputs.is_empty() {
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
                            self.nodes[output.0].inputs.remove(&CircuitNodeIndex(node));
                            self.nodes[output.0].inputs.insert(input);
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
                            self.nodes[output.0].inputs.remove(&CircuitNodeIndex(node));
                        }
                        self.nodes[node].input_distributions.clear();
                        self.nodes[node].outputs.clear();
                    }
                }
            }
                
            // If a distribution node send all its value to a sum node, remove the node from the output
            let mut out_count = (0..self.nodes.len()).map(|_| 0).collect::<Vec<usize>>();
            for node in 0..self.distribution_nodes.len() {
                let number_value = self.get_distribution_domain_size(DistributionNodeIndex(node));
                for (output, _) in self.distribution_nodes[node].outputs.iter().copied() {
                    if !self.nodes[output.0].is_mul {
                        out_count[output.0] += 1;
                    }
                }
                let mut i = 0;
                while i < self.distribution_nodes[node].outputs.len() {
                    let output = self.distribution_nodes[node].outputs[i].0;
                    if out_count[output.0] == number_value {
                        changed = true;
                        self.distribution_nodes[node].outputs.swap_remove(i);
                        self.nodes[output.0].to_remove = true;
                        self.nodes[output.0].input_distributions.clear();
                        for o in 0..self.nodes[output.0].outputs.len() {
                            let o_output = self.nodes[output.0].outputs[o].0;
                            self.nodes[o_output].inputs.remove(&output);
                        }
                    } else {
                        i += 1;
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
    
    // Resets the value of each internal node
    fn reset_nodes(&mut self) {
        for node in self.nodes.iter_mut() {
            if node.is_mul {
                node.value.assign(1.0);
            } else {
                node.value.assign(0.0);
            }
            node.path_value = f128!(1.0);
        }
        for distr in self.distribution_nodes.iter_mut() {
            for val in distr.grad_value.iter_mut() {
                val.assign(0.0);
            }
        }
    }
    
    /// Evaluates the circuits, layer by layer (starting from the input distribution, then layer 0)
    pub fn evaluate(&mut self) -> Float {
        self.reset_nodes();
        for d_node in (0..self.distribution_nodes.len()).map(DistributionNodeIndex) {
            for (output, value) in self.distribution_nodes[d_node.0].outputs.iter().copied() {
                let prob = self.get_distribution_probability_at(d_node, value);
                if self.nodes[output.0].is_mul {
                    self.nodes[output.0].value *= prob;//distribution_nodes[d_node.0].probabilities[value];
                } else {
                    self.nodes[output.0].value += prob;//distribution_nodes[d_node.0].probabilities[value];                    
                }
            }
        }
        for node in (0..self.nodes.len()).map(CircuitNodeIndex) {
            let start = self.nodes[node.0].output_start;
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

    /// Computes the gradient of each distribution, layer by layer (starting from the root, to the distributions)
    pub fn compute_grads(&mut self, grad_loss: f64, lr: f64) {
        for node in (0..self.nodes.len()).map(CircuitNodeIndex).rev() {
            // Iterate on all nodes from the DAC, top-down way
            let start = self.nodes[node.0].input_start;
            let end = start + self.nodes[node.0].number_inputs;
            let value= self.nodes[node.0].value.clone();
            let path_val = self.nodes[node.0].path_value.clone();
            // Update the path value for children that are other nodes
            for i in start..end {
                let child = self.inputs[i];
                if self.nodes[node.0].is_mul {
                    // In case of a multiplicative node, multiply the current value with all the other branches values
                    // == multiply by the node value and divide by the considered branch value
                    self.nodes[child.0].path_value.assign(&path_val);
                    self.nodes[child.0].path_value *=  &value;
                    let child_val = self.nodes[child.0].value.clone();
                    self.nodes[child.0].path_value /= child_val;
                } else {
                    // In case of an additive node, simply propagate the current value
                    self.nodes[child.0].path_value.assign(path_val.clone());                    
                }
            }

            // Compute the gradient for children that are leaf distributions
            for node_i_distr in 0..self.get_circuit_node_number_distribution_input(node){
                let (distr_i, val_i) = self.get_circuit_node_input_distribution_at(node, node_i_distr);
                let mut factor = self.nodes[node.0].path_value.clone()*grad_loss.clone();
                if self.nodes[node.0].is_mul {
                    // In case of a multiplicative node, the factor of the gradient is multiplied by the values of all the other branches of the node
                    factor *= &value;
                    factor /= self.get_distribution_probability_at(DistributionNodeIndex(distr_i.0), val_i);
                }

                // Compute the gradient contribution for the value used in the node and all the other possible values of the distribution
                let mut sum_other_w = f128!(0.0);
                let child_w = self.get_distribution_probability_at(DistributionNodeIndex(distr_i.0), val_i);
                for params in 0..self.get_distribution_domain_size(DistributionNodeIndex(distr_i.0)){
                    let weigth = self.get_distribution_probability_at(DistributionNodeIndex(distr_i.0), params);
                    if params != val_i {
                        // For the other possible values of the distribution, the gradient contribution 
                        // is simply the factor and the product of both weights
                        self.distribution_nodes[distr_i.0].grad_value[params] -= factor.clone()*weigth.clone()*child_w.clone();
                        sum_other_w += weigth.clone();
                    }
                }
                self.distribution_nodes[distr_i.0].grad_value[val_i] += factor.clone()*child_w.clone()*sum_other_w.clone();
            }
        }
        for distr in 0..self.distribution_nodes.len() {
            for val in 0..self.get_distribution_domain_size(DistributionNodeIndex(distr)){
                self.distribution_nodes[distr].unsoftmaxed_probabilities[val] -= lr*self.distribution_nodes[distr].grad_value[val].to_f64();
            }
        }
    }
    
    // --- Various helper methods for the python bindings ---
    

    // --- GETTERS --- //
    
    /// Returns the index of the first output of the node, in the output vector
    pub fn get_circuit_node_out_start(&self, node: CircuitNodeIndex) -> usize {
        self.nodes[node.0].output_start    
    }

    /// Returns the number of output of the node
    pub fn get_circuit_node_number_output(&self, node: CircuitNodeIndex) -> usize {
        self.nodes[node.0].number_outputs    
    }
    
    /// Returns the index of the first input of the node, in the input vector
    pub fn get_circuit_node_in_start(&self, node: CircuitNodeIndex) -> usize {
        self.nodes[node.0].input_start
    }

    /// Returns the number of input of the node
    pub fn get_circuit_node_number_input(&self, node: CircuitNodeIndex) -> usize {
        self.nodes[node.0].number_inputs
    }
    
    /// Returns the probability of the circuit. If the node has not been evaluated,
    /// 1.0 is returned if the root is a multiplication node, and 0.0 if the root is a
    /// sum node
    pub fn get_circuit_node_probability(&self, node: CircuitNodeIndex) -> &Float {
        &self.nodes[node.0].value
    }
    
    /// Returns the probability of the circuit. If the circuit has not been evaluated,
    /// 1.0 is returned if the root is a multiplication node, and 0.0 if the root is a
    /// sum node
    pub fn get_circuit_probability(&self) -> &Float {
        &self.nodes[self.nodes.len()-1].value
    }
    
    /// Returns the node at the given index in the input vector
    pub fn get_input_at(&self, index: usize) -> CircuitNodeIndex {
        self.inputs[index]
    }
    
    /// Returns the node at the given index in the output vector
    pub fn get_output_at(&self, index: usize) -> CircuitNodeIndex {
        self.outputs[index]
    }
    
    /// Returns the number of input edges from the distributions of the given node
    pub fn get_circuit_node_number_distribution_input(&self, node: CircuitNodeIndex) -> usize {
        self.nodes[node.0].input_distributions.len()
    }
    
    /// Returns, for a given node and an index in its distributions input vector, the distribution index of the input and the value
    /// index send from the probability
    pub fn get_circuit_node_input_distribution_at(&self, node: CircuitNodeIndex, index: usize) -> (DistributionIndex, usize) {
        self.nodes[node.0].input_distributions[index]
    }
    
    /// Returns the number of value in the given distribution
    pub fn get_distribution_domain_size(&self, distribution: DistributionNodeIndex) -> usize {
        self.distribution_nodes[distribution.0].unsoftmaxed_probabilities.len()
    }
    
    /// Returns the probability at the given index of the given distribution
    pub fn get_distribution_probability_at(&self, distribution: DistributionNodeIndex, index: usize) -> f64 {
        let softmaxed = softmax(&self.distribution_nodes[distribution.0].unsoftmaxed_probabilities);
        softmaxed[index]
    }
    
    /// Returns the pair (circuit_node, index) for the output of the distribution at the given index
    pub fn get_distribution_output_at(&self, distribution: DistributionNodeIndex, index: usize) -> (CircuitNodeIndex, usize) {
        self.distribution_nodes[distribution.0].outputs[index]
    }
    
    /// Returns the number of output of a distribution node
    pub fn get_distribution_number_output(&self, distribution: DistributionNodeIndex) -> usize {
        self.distribution_nodes[distribution.0].outputs.len()
    }

    /// Returns the gradient of the given distribution at the given index
    pub fn get_distribution_gradient_at(&self, distribution: DistributionNodeIndex, index: usize) -> f64 {
        self.distribution_nodes[distribution.0].grad_value[index].to_f64()
    }
    
    // --- SETTERS --- //
    
    /// Set the unsoftmaxed probability of the distribution, at the given index, to the given value
    pub fn set_distribution_probability_at(&mut self, distribution: DistributionNodeIndex, index: usize, value: f64) {
        self.distribution_nodes[distribution.0].unsoftmaxed_probabilities[index] = value;
    }


    // --- QUERIES --- //

    /// Returns true if the given node is a multiplication node
    pub fn is_circuit_node_mul(&self, node: CircuitNodeIndex) -> bool {
        self.nodes[node.0].is_mul
    }
    
    /// Returns the number of computational nodes in the circuit
    pub fn number_nodes(&self) -> usize {
        self.nodes.len()
    }
    
    /// Returns the number of input distributions in the circuit
    pub fn number_distributions(&self) -> usize {
        self.distribution_nodes.len()
    }

}

// Various methods for dumping the compiled diagram, including standardized format and graphviz (inspired from https://github.com/xgillard/ddo )

// Visualization as graphviz DOT file
impl Dac {
    
    fn distribution_node_id(&self, node: DistributionNodeIndex) -> usize {
        node.0
    }
    
    fn sp_node_id(&self, node: CircuitNodeIndex) -> usize {
        self.distribution_nodes.len() + node.0
    }
    
    pub fn as_graphviz(&self) -> String {
        
        let dist_node_attributes = String::from("shape=circle,style=filled");
        let prod_node_attributes = String::from("shape=circle,style=filled");
        let sum_node_attributes = String::from("shape=circle,style=filled");
        
        let mut out = String::new();
        out.push_str("digraph {\ntranksep = 3;\n\n");
        
        // Generating the nodes in the network 

        for node in (0..self.distribution_nodes.len()).map(DistributionNodeIndex) {
            if !self.distribution_nodes[node.0].outputs.is_empty() {
                let id = self.distribution_node_id(node);
                out.push_str(&Dac::node(id, &dist_node_attributes, &format!("d{}", id)));
            }
        }

        for node in (0..self.nodes.len()).map(CircuitNodeIndex) {
            let id = self.sp_node_id(node);
            if self.nodes[node.0].is_mul {
                out.push_str(&Dac::node(id, &prod_node_attributes, &format!("X ({:.3})", self.nodes[node.0].value)));
            } else {
                out.push_str(&Dac::node(id, &sum_node_attributes, &format!("+ ({:.3})", self.nodes[node.0].value)));
            }
        }
        
        // Generating the edges
        for node in (0..self.distribution_nodes.len()).map(DistributionNodeIndex) {
            let from = self.distribution_node_id(node);
            for (output, value) in self.distribution_nodes[node.0].outputs.iter().copied() {
                let to = self.sp_node_id(output);
                let f_value = format!("({}, {:.3})",value, self.get_distribution_probability_at(node, value));
                out.push_str(&Dac::edge(from, to, Some(f_value)));
            }
        }
        
        for node in (0..self.nodes.len()).map(CircuitNodeIndex) {
            let from = self.sp_node_id(node);
            let start = self.nodes[node.0].output_start;
            let end = start + self.nodes[node.0].number_outputs;
            for output in self.outputs[start..end].iter().copied() {
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
        writeln!(f, "dac {} {}", self.distribution_nodes.len(), self.nodes.len())?;
        for node_i in 0..self.distribution_nodes.len(){
            let dom_size = self.get_distribution_domain_size(DistributionNodeIndex(node_i));
            write!(f, "d {}", dom_size)?;
            for p_i in 0..dom_size{
                write!(f, " {:.8}", self.get_distribution_probability_at(DistributionNodeIndex(node_i), p_i))?;
            }
            let node = &self.distribution_nodes[node_i];
            for (output, value) in node.outputs.iter().copied() {
                write!(f, " {} {}", output.0, value)?;
            }
            writeln!(f)?;
        }
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
            if node.is_mul {
                write!(f, "x")?;       
            } else {
                write!(f, "+")?;
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
            distribution_nodes: vec![],
            outputs: vec![],
            inputs: vec![],
        };
        let file = File::open(filepath).unwrap();
        let reader = BufReader::new(file);
        let mut input_distributions_node: Vec<Vec<(DistributionIndex, usize)>> = vec![];
        for line in reader.lines() {
            let l = line.unwrap();
            let split = l.split_whitespace().collect::<Vec<&str>>();
            if l.starts_with("dac") {
                let values = split.iter().skip(1).map(|i| i.parse::<usize>().unwrap()).collect::<Vec<usize>>();
                let number_nodes = values[1];
                for _ in 0..number_nodes {
                    input_distributions_node.push(vec![]);
                }
            } else if l.starts_with('d') {
                let dom_size = split[1].parse::<usize>().unwrap();
                let mut probabilities = split[2..(2+dom_size)].iter().map(|s| s.parse::<f64>().unwrap()).collect::<Vec<f64>>();
                let mut outputs: Vec<(CircuitNodeIndex, usize)> = vec![];
                for i in ((2+dom_size)..split.len()).step_by(2) {
                    let output_node = CircuitNodeIndex(split[i].parse::<usize>().unwrap());
                    let value = split[i+1].parse::<usize>().unwrap();
                    outputs.push((output_node, value));
                    input_distributions_node[output_node.0].push((DistributionIndex(dac.distribution_nodes.len()), value));
                }
                for el in &mut probabilities {
                    *el = el.log10();
                }
                let prob_len = probabilities.len();
                dac.distribution_nodes.push(DistributionNode {
                    unsoftmaxed_probabilities: probabilities,
                    outputs,
                    grad_value: vec![f128!(0.0); prob_len],
                });
            } else if l.starts_with("inputs") {
                dac.inputs = split.iter().skip(1).map(|i| CircuitNodeIndex(i.parse::<usize>().unwrap())).collect();
            } else if l.starts_with("outputs") {
                dac.outputs = split.iter().skip(1).map(|i| CircuitNodeIndex(i.parse::<usize>().unwrap())).collect();
            } else if l.starts_with('x') || l.starts_with('+') {
                let values = split.iter().skip(1).map(|i| i.parse::<usize>().unwrap()).collect::<Vec<usize>>();
                dac.nodes.push(CircuitNode {
                    value: if l.starts_with('x') { f128!(1.0) } else { f128!(0.0) },
                    outputs: vec![],
                    inputs: FxHashSet::default(),
                    input_distributions: input_distributions_node[dac.nodes.len()].clone(),
                    is_mul: l.starts_with('x'),
                    output_start: values[0],
                    number_outputs: values[1],
                    input_start: values[2],
                    number_inputs: values[3],
                    layer: 0,
                    to_remove: false,
                    path_value: f128!(1.0),
                })
            } else if !l.is_empty() {
                panic!("Bad line format: {}", l);
            }
        }
        dac
    }
}