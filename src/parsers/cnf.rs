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

use super::*;
use crate::core::problem::Problem;
use search_trail::StateManager;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use malachite::rational::Rational;
use crate::common::rational;

pub struct CnfParser {
    input: PathBuf,
    evidence: OsString,
}

impl CnfParser {

    pub fn new(input: PathBuf, evidence: OsString) -> Self {
        Self { input, evidence }
    }
}

impl Parser for CnfParser {

    fn problem_from_file(&self, state: &mut StateManager) -> Problem {
        // First pass to get the distributions
        let distributions = self.distributions_from_file();
        // Second pass to parse the clauses
        let mut clauses: Vec<Vec<isize>> = vec![];
        let file = File::open(&self.input).unwrap();
        let reader = BufReader::new(file);
        for l in reader.lines() {
            match l {
                Err(e) => panic!("Problem while reading file: {}", e),
                Ok(line) => {
                    if !line.starts_with('c') && !line.starts_with('p') {
                        // Note: the space before the 0 is important so that clauses like "1 -10 0" are correctly splitted
                        for clause in line.trim_end().split(" 0").filter(|cl| !cl.is_empty()) {
                            clauses.push(clause.split_whitespace().map(|x| x.parse::<isize>().unwrap()).collect());
                        }
                    }
                }
            }
        }
        let content = evidence_from_os_string(&self.evidence);
        let content = content.split_whitespace().map(|x| x.parse::<isize>().unwrap()).collect::<Vec<isize>>();
        let mut clause: Vec<isize> = vec![];
        for literal in content.iter().copied() {
            if literal == 0 {
                clauses.push(clause.clone());
                clause.clear();
            } else {
                clause.push(literal);
            }
        }
        create_problem(&distributions, &clauses, state)
    }

    fn clauses_from_file(&self) -> Vec<Vec<isize>> {
        // Second pass to parse the clauses
        let mut clauses: Vec<Vec<isize>> = vec![];
        let file = File::open(&self.input).unwrap();
        let reader = BufReader::new(file);
        for l in reader.lines() {
            match l {
                Err(e) => panic!("Problem while reading file: {}", e),
                Ok(line) => {
                    if !line.starts_with('c') && !line.starts_with('p') {
                        // Note: the space before the 0 is important so that clauses like "1 -10 0" are correctly splitted
                        for clause in line.trim_end().split(" 0").filter(|cl| !cl.is_empty()) {
                            clauses.push(clause.split_whitespace().map(|x| x.parse::<isize>().unwrap()).collect());
                        }
                    }
                }
            }
        }
        let content = evidence_from_os_string(&self.evidence);
        let content = content.split_whitespace().map(|x| x.parse::<isize>().unwrap()).collect::<Vec<isize>>();
        let mut clause: Vec<isize> = vec![];
        for literal in content.iter().copied() {
            if literal == 0 {
                clauses.push(clause.clone());
                clause.clear();
            } else {
                clause.push(literal);
            }
        }
        clauses
    }

    
    fn distributions_from_file(&self) -> Vec<Vec<Rational>> {
        let mut distributions: Vec<Vec<Rational>> = vec![];
        let file = File::open(&self.input).unwrap();
        let reader = BufReader::new(&file);
        for l in reader.lines() {
            match l {
                Err(e) => panic!("Problem while parsing the distributions: {}", e),
                Ok(line) => {
                    if line.starts_with("c p distribution") {
                        let weights: Vec<Rational> = line.split_whitespace().skip(3).map(|token| rational(token.parse::<f64>().unwrap())).collect();
                        distributions.push(weights);
                    }
                }
            }
        }
        distributions
    }
}
