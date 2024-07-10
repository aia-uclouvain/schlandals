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

use crate::core::problem::{Problem, DistributionIndex, VariableIndex};
use crate::semiring::*;
use crate::common::Solution;

use super::node::*;
use rug::Float;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex(pub usize);

/// Structure representing the arithmetic circuit.
pub struct Dac<R>
    where R: SemiRing
{
    /// Internal nodes of the circuit
    nodes: Vec<Node<R>>,
    /// Each node has a reference (two usizes) to a slice of this vector to store the index of
    /// their inputs in the circuit
    inputs: Vec<NodeIndex>,
    /// Mapping between the (distribution, value) and the node index for distribution nodes
    distribution_mapping: FxHashMap<(usize, usize), NodeIndex>,
    /// Root of the circuit
    root: Option<NodeIndex>,
    /// Index of the first node that is not an input
    start_computational_nodes: usize,
    /// How much seconds was needed to compile this diagram
    compile_time: u64,
    /// epsilon of the circuit output
    epsilon: f64,
}

impl<R> Dac<R>
    where R: SemiRing
{

    /// Creates a new empty DAC. An input node is created for each distribution in the problem.
    pub fn new(epsilon: f64) -> Self {
        Self {
            nodes: vec![],
            inputs: vec![],
            distribution_mapping: FxHashMap::default(),
            root: None,
            start_computational_nodes: 0,
            compile_time: u64::MAX,
            epsilon,
        }
    }

    pub fn set_compile_time(&mut self, compile_time: u64) {
        self.compile_time = compile_time;
    }

    /// Returns the number of nodes in the circuit
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn is_complete(&self) -> bool {
        self.epsilon.abs() < FLOAT_CMP_THRESHOLD
    }

    /// Adds a prod node to the circuit. Returns its index.
    pub fn prod_node(&mut self, number_children: usize) -> Node<R> {
        let mut node = Node::product();
        node.set_input_start(self.inputs.len());
        node.set_number_inputs(number_children);
        for _ in 0..number_children {
            self.inputs.push(NodeIndex(0));
        }
        node
    }
    
    /// Adds a sum node to the circuit. Returns its index.
    pub fn sum_node(&mut self, number_children: usize) -> Node<R> {
        let mut node = Node::sum();
        node.set_input_start(self.inputs.len());
        node.set_number_inputs(number_children);
        for _ in 0..number_children {
            self.inputs.push(NodeIndex(0));
        }
        node
    }

    pub fn add_node(&mut self, node: Node<R>) -> NodeIndex {
        let id = NodeIndex(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub fn add_input(&mut self, index: usize, node: NodeIndex) {
        self.inputs[index] = node;
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
    
    /// Returns, for a given distribution index and its value, the corresponding node index in the dac
    pub fn distribution_value_node(&mut self, problem: &Problem, distribution: DistributionIndex, variable: VariableIndex) -> NodeIndex {
        let distribution_index = problem[distribution].old_index();
        let value_index = problem[variable].index_in_distribution().unwrap();
        let weight = problem[variable].weight().unwrap();
        if let Some(x) = self.distribution_mapping.get(&(distribution_index, value_index)) {
            *x
        } else {
            self.nodes.push(Node::distribution(distribution_index, value_index, weight));
            self.distribution_mapping.insert((distribution_index, value_index), NodeIndex(self.nodes.len()-1));
            NodeIndex(self.nodes.len()-1)
        }
    }
}

// --- CIRCUIT EVALUATION ---

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
        let p = self.nodes.last().unwrap().value().to_f64();
        Solution::new(F128!(p), F128!(p), self.compile_time)
    }

    /// Updates the values of the distributions to the given values
    pub fn reset_distributions(&mut self, distributions: &[Vec<R>]) {
        // TODO: Stop after the last distribution
        for node in (0..self.nodes.len()).map(NodeIndex) {
            if let NodeType::Distribution { d, v } = self[node].get_type() {
                let value = R::copy(&distributions[d][v]);
                self[node].set_value(value);
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
                let value = if start != end { R::mul_children((start..end).map(|idx| {
                    let child = self.inputs[idx];
                    self[child].value()
                })) } else { R::zero() };
                self[node].set_value(value);
            } else if self[node].is_sum() {
                let value = if start != end { R::sum_children((start..end).map(|idx| {
                    let child = self.inputs[idx];
                    self[child].value()
                })) } else { R::one() };
                self[node].set_value(value);
            }
        }
        // Last node is the root since it has the higher layer
        self.nodes.last().unwrap().value()
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

        let mut out = String::new();
        out.push_str("digraph {\ntranksep = 3;\n\n");

        // Generating the nodes in the network 
        for node in (0..self.nodes.len()).map(NodeIndex) {
            let id = node.0;
            let value = format!("{:.4}", self[node].value().to_f64());
            match self[node].get_type() {
                NodeType::Sum => {
                    let attributes = &sum_node_attributes;
                    let label = format!("+ {}", value);
                    out.push_str(&format!("\t{id} [{attributes},label=\"{label}\"];\n"));
                },
                NodeType::Product => {
                    let attributes = &prod_node_attributes;
                    let label = format!("x {}", value);
                    out.push_str(&format!("\t{id} [{attributes},label=\"{label}\"];\n"));
                },
                NodeType::Distribution{d ,v } => {
                    let attributes = &dist_node_attributes;
                    let label = format!("D {} (d{} v{})", value, d, v);
                    out.push_str(&format!("\t{id} [{attributes},label=\"{label}\"];\n"));
                },
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

}

// Custom text format for the network

impl<R> fmt::Display for Dac<R>
where R: SemiRing
{
    /// Formats the circuit in the following formats
    ///     1. The first line starts with "input" followed by the indexes in the inputs vector,
    ///        separated by a space
    ///     2. There is one line per node in the diagram, starting with "x" for product node, "+"
    ///        for sum nodes, "a" for approximate node and "d" for distribution node. For
    ///        approximate and distribution node, their parameters are written next to them.
    ///        Then, for all types of node, their slice (start and size) in the inputs
    ///        vector are written
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f)?;
        write!(f, "inputs")?;
        for input in self.inputs.iter().copied().map(|node| format!("{}", node.0)) {
            write!(f, " {}", input)?;
        }
        writeln!(f)?;
        for node in self.nodes.iter() {
            match node.get_type() {
                NodeType::Product => write!(f, "x")?,
                NodeType::Sum => write!(f, "+")?,
                NodeType::Distribution {d, v} => write!(f, "d {} {}", d, v)?,
            }
            writeln!(f, " {} {}", node.input_start(), node.number_inputs())?;
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
            inputs: vec![],
            distribution_mapping: FxHashMap::default(),
            root: None,
            start_computational_nodes: 0,
            compile_time: 0,
            epsilon: 0.0,
        };
        let file = File::open(filepath).unwrap();
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let l = line.unwrap();
            let split = l.split_whitespace().collect::<Vec<&str>>();
            if l.starts_with("inputs") {
                dac.inputs = split.iter().skip(1).map(|i| NodeIndex(i.parse::<usize>().unwrap())).collect();
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
                node.set_input_start(values[2]);
                node.set_number_inputs(values[3]);
                dac.nodes.push(node);
            } else if l.starts_with('d') {
                let values = split.iter().skip(1).map(|i| i.parse::<usize>().unwrap()).collect::<Vec<usize>>();
                let d = values[0];
                let v = values[1];
                let mut node: Node<Float> = Node::distribution(d, v, 0.5);
                node.set_input_start(values[4]);
                node.set_number_inputs(values[5]);
                dac.distribution_mapping.insert((d, v), NodeIndex(dac.nodes.len()-1));
            }
            else if !l.is_empty() {
                panic!("Bad line format: {}", l);
            }
        }
        dac
    }
}

impl<R> Default for Dac<R>
    where R: SemiRing
{
    fn default() -> Self {
        Self::new(0.0)
    }
}
