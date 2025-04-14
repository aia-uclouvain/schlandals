//! This module provide an implementation of a node in an arithmetic circuits.

use malachite::rational::Rational;
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
pub struct Node {
    /// Value of the node. Initialized at 1.0 for product, at 0.0 for sum and at a specific value for distribution nodes.
    /// For product and sum nodes, after the evaluation it is equal to the product (sum) of its input values.
    value: Rational,
    /// Type of node
    nodetype: NodeType,
    /// Start of the inputs in the vector of inputs
    input_start: usize,
    /// Number of inputs the node has
    number_inputs: usize,
    /// The multiplicative factor accumulated on the paths to the root whil computing the gradient
    /// (only used when evaluating on the Rational semiring)
    path_value: Rational,
}

impl Node {
    /// Returns a new product node
    pub fn product() -> Self {
        Node {
            value: rational(1.0),
            nodetype: NodeType::Product,
            input_start: 0,
            number_inputs: 0,
            path_value: rational(1.0),
        }
    }

    /// Returns a new sum node
    pub fn sum() -> Self {
        Node {
            value: rational(0.0),
            nodetype: NodeType::Sum,
            input_start: 0,
            number_inputs: 0,
            path_value: rational(1.0),
        }
    }

    /// Returns a new distribution node with P[distribution = value] = probability
    pub fn distribution(distribution: usize, value: usize, probability: Rational) -> Self {
        Node {
            value: probability,
            nodetype: NodeType::Distribution {d: distribution, v: value},
            input_start: 0,
            number_inputs: 0,
            path_value: rational(1.0),
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
    pub fn value(&self) -> &Rational {
        &self.value
    }

    /// Returns a mutable reference to the value stored in the node
    pub fn value_mut(&mut self) -> &mut Rational {
        &mut self.value
    }

    // Return the path value of the node. The path value of a node is the accumulated product of
    // the value of the nodes from the root of the circuit to the node.
    pub fn path_value(&self) -> Rational {
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
    pub fn set_value(&mut self, value: Rational){
        self.value = value
    }

    pub fn assign(&mut self, value: &Rational) {
        self.value = value.clone();
    }

    /// Sets the path value of the node to the given float
    pub fn set_path_value(&mut self, value: Rational){
        self.path_value = value;
    }

    /// Adds the given float to the path value of the node
    pub fn add_to_path_value(&mut self, value: Rational){
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
