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

//! This module provides a parser for a custom DIMACS format used by our solver. It is called
//! PPIDIMACS for "Positive Probabilistic Implications DIMACS".
//! An example of valid file is given next
//!
//! c This line is a comment
//! c We define a problem in cfn form with 7 variables, 3 clauses and 2 probabilistic variables
//! p cfn 7 3
//! c This define the probabilistic variables as well as their weights
//! c A line starting with d means that we define a distribution.
//! c The line of a distribution must sum up to 1
//! c A distribution is a succession of pair variable-weight
//! c The indexe of the distribution are consecutive, the following distribution has two nodes
//! c indexed 0 and 1
//! d 0.3 0.7
//! c Nodes with index 2 and 3
//! d 0.4 0.6
//! c This define the clauses as in the DIMACS-cfn format
//! c This clause is 0 and 5 => 4
//! 4 -0 -5
//! 5 -1 -2
//! 6 -3 -4
//!     
//! The following restrictions are imposed on the clauses
//!     1. All clauses must be implications with positive literals. This means that in CFN the
//!        clauses have exactly one positive literals which is the head of the implication. All
//!        variable appearing in the implicant must be negated.
//!     2. The head of the implications can not be a probabilistic variable

use crate::core::graph::{Graph, NodeIndex};
use crate::core::trail::StateManager;
use crate::solver::propagator::SimplePropagator;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

pub fn graph_from_ppidimacs<S: StateManager>(filepath: &PathBuf, state: &mut S) -> Graph {
    let mut node_to_propagate: Vec<(NodeIndex, bool)> = Vec::new();
    let mut g = Graph::new(state);
    let file = File::open(filepath).unwrap();
    let reader = BufReader::new(file);
    let mut number_nodes: Option<usize> = None;
    let mut distribution_definition_finished = false;
    let mut line_count = 0;
    for line in reader.lines() {
        let l = line.unwrap();
        if l.starts_with("c") {
            continue;
        }
        if l.starts_with("p cfn") {
            // Header, parse the number of clauses and variables
            let mut split = l.split_whitespace();
            number_nodes = Some(split.nth(2).unwrap().parse::<usize>().unwrap());
        } else if l.starts_with("d") {
            if distribution_definition_finished {
                panic!("[Parsing error at line {}] All distribution should be defined before the clauses", line_count);
            }
            let split = l
                .split_whitespace()
                .skip(1)
                .map(|token| token.parse::<f64>().unwrap().log2())
                .collect::<Vec<f64>>();
            g.add_distribution(&split, state);
        } else {
            // First line for the clauses
            if number_nodes.is_none() {
                panic!("[Parsing error at line {}] The head ``p cfn n m`` is not defined before the clauses", line_count);
            }
            if !distribution_definition_finished {
                distribution_definition_finished = true;
                let current_number_of_nodes = g.number_nodes();
                for _ in current_number_of_nodes..number_nodes.unwrap() {
                    g.add_node(false, None, None, state);
                }
            }
            let split = l.split_whitespace().collect::<Vec<&str>>();
            let positive_literals = split
                .iter()
                .filter(|x| !x.starts_with("-"))
                .map(|x| x.parse::<usize>().unwrap())
                .collect::<Vec<usize>>();
            let negative_literals = split
                .iter()
                .filter(|x| x.starts_with("-"))
                .map(|x| x.parse::<isize>().unwrap() * -1)
                .collect::<Vec<isize>>();
            if positive_literals.len() > 1 {
                panic!("[Parsing error at line {}] There are more than one positive literals in this clause", line_count);
            }
            let head = if positive_literals.len() == 0 {
                // There is no head in this clause, so it is just a clause of the form
                //      n1 && n2 && ... && nn =>
                //  which, in our model implies that the head is false (otherwise it does not
                //  constrain the problem)
                let n = g.add_node(false, None, None, state);
                node_to_propagate.push((n, false));
                n
            } else {
                NodeIndex(positive_literals[0])
            };
            let body = negative_literals
                .iter()
                .copied()
                .map(|x| NodeIndex(x as usize))
                .collect::<Vec<NodeIndex>>();
            g.add_clause(head, &body, state);
        }
        line_count += 1;
    }
    for (node, value) in node_to_propagate {
        g.propagate_node(node, value, state);
    }
    g.propagate(state);
    g
}

#[cfg(test)]
mod test_ppidimacs_parsing {

    use super::graph_from_ppidimacs;
    use crate::core::graph::NodeIndex;
    use crate::core::trail::TrailedStateManager;
    use std::path::PathBuf;

    #[test]
    fn test_file() {
        let mut file = PathBuf::new();
        let mut state = TrailedStateManager::new();
        file.push("tests/instances/bayesian_networks/abc_chain_b0.ppidimacs");
        let g = graph_from_ppidimacs(&file, &mut state);
        // Nodes for the distributions, the deterministics + 1 node for the vb0 -> False
        assert_eq!(17, g.number_nodes());
        assert_eq!(5, g.number_distributions());

        let nodes: Vec<NodeIndex> = g.nodes_iter().collect();
        for i in 0..10 {
            assert!(g.is_node_probabilistic(nodes[i]));
        }
    }
}
