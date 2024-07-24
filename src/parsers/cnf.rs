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

use super::create_problem;
use crate::core::problem::Problem;
use search_trail::StateManager;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;


pub fn problem_from_cnf(
    filepath: &PathBuf,
    state: &mut StateManager,
) -> Problem {
    // First pass to get the distributions
    let distributions = distributions_from_cnf(filepath);
    // Second pass to parse the clauses
    let mut clauses: Vec<Vec<isize>> = vec![];
    let file = File::open(filepath).unwrap();
    let reader = BufReader::new(file);
    for l in reader.lines() {
        match l {
            Err(e) => panic!("Problem while reading file: {}", e),
            Ok(line) => {
                if !line.starts_with('c') && !line.starts_with('p') {
                    // Note: the space before the 0 is important so that clauses like "1 -10 0" are correctly splitted
                    for clause in line.split(" 0").filter(|cl| !cl.is_empty()) {
                        clauses.push(clause.split_whitespace().map(|x| x.parse::<isize>().unwrap()).collect());
                    }
                }
            }
        }
    }
    create_problem(&distributions, &clauses, state)
}

pub fn distributions_from_cnf(filepath: &PathBuf) -> Vec<Vec<f64>> {
    let mut distributions: Vec<Vec<f64>> = vec![];
    let file = File::open(filepath).unwrap();
    let reader = BufReader::new(&file);
    for l in reader.lines() {
        match l {
            Err(e) => panic!("Problem while parsing the distributions: {}", e),
            Ok(line) => {
                if line.starts_with("c p distribution") {
                    let weights: Vec<f64> = line.split_whitespace().skip(3).map(|token| token.parse::<f64>().unwrap()).collect();
                    distributions.push(weights);
                }
            }
        }
    }
    distributions
}

pub fn learned_distributions_from_cnf(filepath: &PathBuf) -> Vec<bool> {
    let mut number_distributions = 0;
    let mut learned_distributions: Vec<usize> = vec![];
    let file = File::open(filepath).unwrap();
    let reader = BufReader::new(&file);
    for l in reader.lines() {
        match l {
            Err(e) => panic!("Problem while parsing the learned distributions: {}", e),
            Ok(line) => {
                if line.starts_with("c p distribution") {
                    number_distributions += 1;
                } else if line.starts_with("c p learn") {
                    for d in line.split_whitespace().skip(3).map(|s| s.parse::<usize>().unwrap() - 1) {
                        learned_distributions.push(d);
                    }
                }
            }
        }
    }
    let mut flags = if learned_distributions.is_empty() {vec![true; number_distributions]} else {vec![false; number_distributions]};
    for x in learned_distributions {
        flags[x] = true;
    }
    flags
}
