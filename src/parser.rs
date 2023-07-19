//Schlandals
//Copyright (C) 2022-2023 A. Dubray
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

use crate::core::graph::{Graph, VariableIndex};
use crate::propagator::FTReachablePropagator;
use search_trail::StateManager;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

pub fn graph_from_ppidimacs<const C: bool>(
    filepath: &PathBuf,
    state: &mut StateManager,
    propagator: &mut FTReachablePropagator<C>,
) -> Graph {
    let mut g = Graph::new(state);
    let file = File::open(filepath).unwrap();
    let reader = BufReader::new(file);
    let mut number_nodes: Option<usize> = None;
    let mut distribution_definition_finished = false;
    let mut line_count = 0;
    for line in reader.lines() {
        let l = line.unwrap();
        if l.starts_with("p cnf") {
            // Header, parse the number of clauses and variables
            let mut split = l.split_whitespace();
            number_nodes = Some(split.nth(2).unwrap().parse::<usize>().unwrap());
        } else if l.starts_with("c p distribution") {
            if distribution_definition_finished {
                panic!("[Parsing error at line {}] All distribution should be defined before the clauses", line_count);
            }
            let split = l
                .split_whitespace()
                .skip(3)
                .map(|token| token.parse::<f64>().unwrap())
                .collect::<Vec<f64>>();
            let nodes = g.add_distribution(&split, state);
            for i in 0..split.len() {
                if !C && split[i] == 1.0 {
                    propagator.add_to_propagation_stack(nodes[i], true);
                }
            }
        } else if l.starts_with('c'){
            continue;
        } else {
            // First line for the clauses
            if number_nodes.is_none() {
                panic!("[Parsing error at line {}] The head ``p cnf n m`` is not defined before the clauses", line_count);
            }
            if !distribution_definition_finished {
                distribution_definition_finished = true;
                let current_number_of_nodes = g.number_variables();
                for _ in current_number_of_nodes..number_nodes.unwrap() {
                    g.add_variable(false, None, None, state);
                }
            }
            // Note: the space before the 0 is important so that clauses like "1 -10 0" are correctly splitted
            for clause in l.split(" 0").filter(|cl| !cl.is_empty()) {
                let split = clause.split_whitespace().collect::<Vec<&str>>();
                let positive_literals = split
                    .iter()
                    .filter(|x| !x.starts_with('-'))
                    .map(|x| x.parse::<usize>().unwrap() - 1)
                    .collect::<Vec<usize>>();
                let negative_literals = split
                    .iter()
                    .filter(|x| x.starts_with('-'))
                    .map(|x| (-x.parse::<isize>().unwrap()) - 1)
                    .collect::<Vec<isize>>();
                if positive_literals.len() > 1 {
                    panic!("[Parsing error at line {}] There are more than one positive literals in this clause", line_count);
                }
                let head = if positive_literals.is_empty() {
                    // There is no head in this clause, so it is just a clause of the form
                    //      n1 && n2 && ... && nn =>
                    //  which, in our model implies that the head is false (otherwise it does not
                    //  constrain the problem)
                    let n = g.add_variable(false, None, None, state);
                    propagator.add_to_propagation_stack(n, false);
                    n
                } else {
                    VariableIndex(positive_literals[0])
                };
                let body = if negative_literals.is_empty() {
                    let n = g.add_variable(false, None, None, state);
                    propagator.add_to_propagation_stack(n, true);
                    vec![n]
                } else {
                    negative_literals
                        .iter()
                        .copied()
                        .map(|x| VariableIndex(x as usize))
                        .collect::<Vec<VariableIndex>>()
                };
                g.add_clause(head, body, state);
            }
        }
        line_count += 1;
    }
    g
}
/*
#[cfg(test)]
mod test_ppidimacs_parsing {

    use super::graph_from_ppidimacs;
    use crate::core::graph::VariableIndex;
    use crate::core::trail::StateManager;
    use std::path::PathBuf;

    #[test]
    fn test_file() {
        let mut file = PathBuf::new();
        let mut state = StateManager::default();
        file.push("tests/instances/bayesian_networks/abc_chain_b0.ppidimacs");
        let (g, _) = graph_from_ppidimacs(&file, &mut state);
        // Nodes for the distributions, the deterministics + 1 node for the vb0 -> False
        assert_eq!(17, g.number_nodes());
        assert_eq!(5, g.number_distributions());

        let nodes: Vec<VariableIndex> = g.nodes_iter().collect();
        for i in 0..10 {
            assert!(g.is_node_probabilistic(nodes[i]));
        }
    }
}
*/