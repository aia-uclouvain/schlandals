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

use rustc_hash::FxHashSet;

use rug::Float;
use crate::core::graph::{DistributionIndex, VariableIndex, ClauseIndex};
use crate::diagrams::semiring::*;
use super::dac::NodeIndex;
use crate::common::*;
use bitvec::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeNode {
    Product,
    Sum,
    Distribution {d: usize, v: usize},
}

macro_rules! is_node_type {
    ($val:expr, $var:path) => {
        match $val {
            $var{..} => true,
            _ => false,
        }
    }
}

impl TypeNode {
    pub fn get_value(&self) -> usize {
        match self {
            TypeNode::Product => panic!("Product typenode has no value"),
            TypeNode::Sum => panic!("Sum typenode has no value"),
            TypeNode::Distribution{d:_,v} => *v,
        }
    }
}

/// A node structur that represents both internal and distribution nodes.
pub struct Node<R>
    where R: SemiRing
{
    /// Value of the node. Initialized at 1.0 for product, at 0.0 for sum and at a specific value for distribution nodes.
    /// For product and sum nodes, after the evaluation it is equal to the product (sum) of its input values.
    value: R,
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
    /// Gradient computation, the value of the path from the root
    path_value: Float,
    /// Propagation path
    propagation: Vec<(VariableIndex, bool)>,
    clauses: Vec<ClauseIndex>,
    /// If the node has been partially compiled, store the distributions it contains in the
    /// sub-problem it represent
    distributions: Option<BitVec>,
    is_unsat: bool,
    bounding_factor: f64,
}

impl<R> Node<R>
    where R: SemiRing
{
    /// Returns a new product node
    pub fn product() -> Self {
        Node {
            value: R::one(),
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
            propagation: vec![],
            clauses: vec![],
            distributions: None,
            is_unsat: false,
            bounding_factor: 1.0,
        }
    }

    pub fn sum() -> Self {
        Node {
            value: R::zero(),
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
            propagation: vec![],
            clauses: vec![],
            distributions: None,
            is_unsat: false,
            bounding_factor: 1.0,
        }
    }

    pub fn distribution(distribution: usize, value: usize, probability: f64) -> Self {
        Node {
            value: R::from_f64(probability),
            outputs: vec![],
            inputs: FxHashSet::default(),
            typenode: TypeNode::Distribution {d: distribution, v: value},
            output_start: 0,
            number_outputs: 0,
            input_start: 0,
            number_inputs: 0,
            layer: 0,
            to_remove: true,
            path_value: f128!(1.0),
            propagation: vec![],
            clauses: vec![],
            distributions: None,
            is_unsat: false,
            bounding_factor: 1.0
        }
    }

    /// Returns true iff the node is a distribution node
    pub fn is_distribution(&self) -> bool {
        is_node_type!(self.typenode, TypeNode::Distribution)
    }

    /// Returns true iff the node is a product node
    pub fn is_product(&self) -> bool {
        is_node_type!(self.typenode, TypeNode::Product)
    }

    /// Returns true iff the node is a sum node
    pub fn is_sum(&self) -> bool {
        is_node_type!(self.typenode, TypeNode::Sum)
    }

    /// Returns the value of the node
    pub fn get_value(&self) -> &R {
        &self.value
    }

    pub fn get_value_mut(&mut self) -> &mut R {
        &mut self.value
    }

    // Return the path value of the node. The path value of a node is the accumulated product of
    // the value of the nodes from the root of the circuit to the node.
    pub fn get_path_value(&self) -> Float{
        self.path_value.clone()
    }

    /// Returns the type of the node.
    pub fn get_type(&self) -> TypeNode{
        self.typenode
    }
    
    /// Returns the start of the outputs of the nodes
    pub fn get_output_start(&self) -> usize {
        self.output_start
    }
    
    /// Returns the number of output the node has
    pub fn get_number_outputs(&self) -> usize {
        self.number_outputs
    }

    /// Returns the start of the input of the node
    pub fn get_input_start(&self) -> usize {
        self.input_start
    }

    /// Returns the number of input the node has
    pub fn get_number_inputs(&self) -> usize {
        self.number_inputs
    }

    /// Returns the layer of the node
    pub fn get_layer(&self) -> usize {
        self.layer
    }

    /// Returns true if the node must be removed
    pub fn is_to_remove(&self) -> bool {
        self.to_remove
    }

    pub fn is_unsat(&self) -> bool {
        self.is_unsat
    }

    /// Returns the propagations that need to be done to reach the node
    pub fn get_propagation(&self) -> &Vec<(VariableIndex, bool)> {
        &self.propagation
    }

    /// Adds the pair (variable, value) to te propagation stack
    pub fn add_to_propagation(&mut self, variable: VariableIndex, value: bool) {
        self.propagation.push((variable, value));
    }

    pub fn add_to_clauses(&mut self, clause: ClauseIndex) {
        self.clauses.push(clause);
    }

    pub fn get_clauses(&self) -> &Vec<ClauseIndex> {
        &self.clauses
    }

    pub fn clear_incomplete(&mut self) {
        self.propagation.clear();
        self.clauses.clear();
    }
    /// Returns true iff the node is incomplete. A node is incomplete if the compilation has been
    /// stopped while the sub-problem represented by the node was not solved. In that case, the
    /// propagations to reach the node are stored in the `propagation field`.
    pub fn is_node_incomplete(&self) -> bool {
        !self.propagation.is_empty()
    }

    /// Adds the distributions to the pool of distribution in the partial node
    pub fn add_distributions(&mut self, number_distributions: usize, distributions: impl Iterator<Item = DistributionIndex>) {
        let mut bv = BitVec::new();
        for _ in 0..number_distributions {
            bv.push(false);
        }
        for d in distributions {
            *bv.get_mut(d.0).unwrap() = true;
        }
        self.distributions = Some(bv);
    }

    /// Returns true if the node is partial and can branch on the given distribution, false
    /// otherwise
    pub fn has_distribution(&self, distribution: DistributionIndex) -> bool {
        match &self.distributions {
            Some(bv) => {
                bv[distribution.0]
            },
            None => false,
        }
    }

    /// Returns true iff the node has some output
    pub fn has_output(&self) -> bool {
        self.number_outputs > 0
    }

    // --- Setters --- /

    /// Adds the given node to the outputs
    pub fn add_output(&mut self, node: NodeIndex) {
        self.outputs.push(node);
    }

    /// Adds the given node to the inputs
    pub fn add_input(&mut self, node: NodeIndex) {
        self.inputs.insert(node);
    }

    /// Sets the value of the node to the given float
    pub fn set_value(&mut self, value: R){
        self.value = value
    }

    pub fn assign(&mut self, value: &R) {
        self.value.set_value(value);
    }

    /// Sets the path value of the node to the given float
    pub fn set_path_value(&mut self, value: Float){
        self.path_value = value;
    }

    /// Adds the given float to the path value of the node
    pub fn add_to_path_value(&mut self, value: Float){
        self.path_value += value;
    }

    /// Sets the type of the node
    pub fn set_type(&mut self, typenode: TypeNode){
        self.typenode = typenode;
    }

    /// Sets the start of the output of the node
    pub fn set_output_start(&mut self, output_start: usize){
        self.output_start = output_start;
    }

    pub fn set_unsat(&mut self) {
        self.is_unsat = true;
    }

    pub fn get_bounding_factor(&self) -> f64 {
        self.bounding_factor
    }

    pub fn set_bounding_factor(&mut self, factor: f64) {
        self.bounding_factor = factor;
    }

    /// Sets the number of output of the node
    pub fn set_number_outputs(&mut self, number_outputs: usize){
        self.number_outputs = number_outputs;
    }

    /// Increments by one the number of outputs of the nodes
    pub fn increment_number_output(&mut self) {
        self.number_outputs += 1;
    }

    /// Increments by one the number of inputs of the nodes
    pub fn increment_number_input(&mut self) {
        self.number_inputs += 1;
    }

    /// Sets the start of the input of the node
    pub fn set_input_start(&mut self, input_start: usize){
        self.input_start = input_start;
    }

    /// Sets the number of input of the node
    pub fn set_number_inputs(&mut self, number_inputs: usize){
        self.number_inputs = number_inputs;
    }

    /// Sets the layer of the node
    pub fn set_layer(&mut self, layer: usize){
        self.layer = layer;
    }

    /// Sets the node to be removed
    pub fn set_to_remove(&mut self, to_remove: bool) {
        self.to_remove = to_remove;
    }

    /// Clear and shrink the output vector
    pub fn clear_and_shrink_output(&mut self) {
        self.outputs.clear();
        self.outputs.shrink_to_fit();
    }

    /// Clear and shrink the input vector
    pub fn clear_and_shrink_input(&mut self) {
        self.inputs.clear();
        self.inputs.shrink_to_fit();
    }

    /// Remove a node at a particular index from the outputs
    pub fn remove_index_from_output(&mut self, index: usize) {
        self.outputs.swap_remove(index);
    }

    /// Removes a node from the inputs
    pub fn remove_input(&mut self, input: NodeIndex) {
        self.inputs.remove(&input);
    }

    pub fn get_output_at(&self, id: usize) -> NodeIndex {
        self.outputs[id]
    }

    // ---- ITERATORS --- //

    /// Returns an iterator over the outputs of the node
    pub fn iter_output(&self) -> impl Iterator<Item = usize> {
        0..self.outputs.len()
    }

    /// Returns an interator over the inputs of the nodes
    pub fn iter_input(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.inputs.iter().copied()
    }

}
