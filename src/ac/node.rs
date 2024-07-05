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

//! This module provide an implementation of a node in an arithmetic circuits.
//! The node is generic over a semiring R

use rug::Float;
use crate::ac::semiring::*;
use crate::common::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Types of node in an AC
pub enum NodeType {
    /// Product nodes
    Product,
    /// Sum nodes
    Sum,
    /// Distribution node. Send the value P[d = v] as output and act as input of the circuit
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

/// A node structure that represents both internal and distribution nodes.
pub struct Node<R>
    where R: SemiRing
{
    /// Value of the node. Initialized at 1.0 for product, at 0.0 for sum and at a specific value for distribution nodes.
    /// For product and sum nodes, after the evaluation it is equal to the product (sum) of its input values.
    value: R,
    /// Type of node
    nodetype: NodeType,
    /// Start of the inputs in the vector of inputs
    input_start: usize,
    /// Number of inputs the node has
    number_inputs: usize,
    /// The multiplicative factor accumulated on the paths to the root whil computing the gradient
    /// (only used when evaluating on the Float semiring)
    path_value: Float,
}

impl<R> Node<R>
    where R: SemiRing
{
    /// Returns a new product node
    pub fn product() -> Self {
        Node {
            value: R::one(),
            nodetype: NodeType::Product,
            input_start: 0,
            number_inputs: 0,
            path_value: F128!(1.0),
        }
    }

    /// Returns a new sum node
    pub fn sum() -> Self {
        Node {
            value: R::zero(),
            nodetype: NodeType::Sum,
            input_start: 0,
            number_inputs: 0,
            path_value: F128!(1.0),
        }
    }

    /// Returns a new distribution node with P[distribution = value] = probability
    pub fn distribution(distribution: usize, value: usize, probability: f64) -> Self {
        Node {
            value: R::from_f64(probability),
            nodetype: NodeType::Distribution {d: distribution, v: value},
            input_start: 0,
            number_inputs: 0,
            path_value: F128!(1.0),
        }
    }

    /// Returns true iff the node is a distribution node
    pub fn is_distribution(&self) -> bool {
        is_node_type!(self.nodetype, NodeType::Distribution)
    }

    /// Returns true iff the node is a product node
    pub fn is_product(&self) -> bool {
        is_node_type!(self.nodetype, NodeType::Product)
    }

    /// Returns true iff the node is a sum node
    pub fn is_sum(&self) -> bool {
        is_node_type!(self.nodetype, NodeType::Sum)
    }

    /// Returns a reference to the value stored in the node
    pub fn value(&self) -> &R {
        &self.value
    }

    /// Returns a mutable reference to the value stored in the node
    pub fn value_mut(&mut self) -> &mut R {
        &mut self.value
    }

    // Return the path value of the node. The path value of a node is the accumulated product of
    // the value of the nodes from the root of the circuit to the node.
    pub fn path_value(&self) -> Float {
        self.path_value.clone()
    }

    /// Returns the type of the node.
    pub fn get_type(&self) -> NodeType {
        // can not name the function "type" since it's a reserved keyword:(
        self.nodetype
    }
    
    /// Returns the start of the input of the node
    pub fn input_start(&self) -> usize {
        self.input_start
    }

    /// Returns the number of input the node has
    pub fn number_inputs(&self) -> usize {
        self.number_inputs
    }

    // --- Setters --- /

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
    pub fn set_type(&mut self, nodetype: NodeType){
        self.nodetype = nodetype;
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
}
