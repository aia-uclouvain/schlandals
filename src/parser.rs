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

use crate::core::problem::{Problem, VariableIndex, DistributionIndex};
use super::core::literal::Literal;
use search_trail::StateManager;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

pub enum FileType {
    CNF,
    FDAC,
}

pub fn problem_from_problem(distributions: &Vec<Vec<f64>>, clauses: &Vec<Vec<isize>>, state: &mut StateManager) -> Problem {
    let mut number_var = 0;
    for clause in clauses.iter() {
        number_var = number_var.max(clause.iter().map(|l| l.abs() as usize).max().unwrap());
    }
    let mut g = Problem::new(state, number_var, clauses.len());
    let variable_mapping = g.add_distributions(distributions, state);
    for clause in clauses.iter() {
        let mut literals: Vec<Literal> = vec![];
        let mut head: Option<Literal> = None;
        for lit in clause.iter().copied() {
            if lit == 0 {
                panic!("Variables in clauses can not be 0");
            }
            let mut variable = lit.abs() as usize;
            if let Some(v) = variable_mapping.get(&variable) {
                variable = *v;
            }
            let var = VariableIndex(variable - 1);
            let trail_value_index = g[var].get_value_index();
            let literal = Literal::from_variable(var, lit > 0, trail_value_index);
            if lit > 0 {
                if head.is_some() {
                    panic!("The clauses {} has more than one positive literal", clause.iter().map(|i| format!("{}", i)).collect::<Vec<String>>().join(" "));
                }
                head = Some(literal);
            }
            literals.push(literal);
        }
        g.add_clause(literals, head, state, false);
    }
    g
}

fn parse_unweighted_problem(filepath: &PathBuf, state: &mut StateManager) -> Problem {
    let file = File::open(filepath).unwrap();
    let reader = BufReader::new(&file);
    let mut number_var = 0;
    let mut number_clauses = 0;
    for l in reader.lines() {
        match l {
            Err(e) => panic!("Error while reading line from input: {}", e),
            Ok(line) => {
                if line.starts_with("p cnf") {
                    let mut split_header = line.split_whitespace();
                    number_var = split_header.nth(2).unwrap().parse::<usize>().unwrap();
                    number_clauses = split_header.next().unwrap().parse::<usize>().unwrap();
                    break;
                }
            }
        }
    }

    let distributions: Vec<Vec<f64>> = (0..number_var).map(|_| vec![0.5, 0.5]).collect();
    let mut problem = Problem::new(state, number_var*2, number_clauses);
    let variable_mapping = problem.add_distributions(&distributions, state);
    let file = File::open(filepath).unwrap();
    let reader = BufReader::new(file);
    for l in reader.lines() {
        match l {
            Err(e) => panic!("Problem while reading file: {}", e),
            Ok(line) => {
                if !line.starts_with('c') && !line.starts_with('p') {
                    // Note: the space before the 0 is important so that clauses like "1 -10 0" are correctly splitted
                    for clause in line.split(" 0").filter(|cl| !cl.is_empty()) {
                        let mut literals: Vec<Literal> = vec![];
                        for lit in clause.split_whitespace() {
                            let parsed_lit = lit.parse::<isize>().unwrap();
                            let is_positive = parsed_lit > 0;
                            // Why did I do this ?
                            //let mut variable = (parsed_lit.abs() * 2 - if is_positive { 0 } else { 1 }) as usize;
                            let mut variable = parsed_lit.abs() as usize;
                            if let Some(new_var) = variable_mapping.get(&variable) {
                                variable = *new_var;
                            }
                            let trail_value_index = problem[VariableIndex(variable - 1)].get_value_index();
                            let literal = Literal::from_variable(VariableIndex(variable - 1), false, trail_value_index);
                            literals.push(literal);
                        }
                        problem.add_clause(literals, None, state, false);
                    }
                }
            }
        }
    }
    problem
}

pub fn problem_from_cnf(
    filepath: &PathBuf,
    state: &mut StateManager,
    learn: bool,
    unweighted: bool,
) -> Problem {
    if unweighted {
        return parse_unweighted_problem(filepath, state);
    }
    // First pass to get the distributions
    let distributions = distributions_from_cnf(filepath);
    let file = File::open(filepath).unwrap();
    let mut reader = BufReader::new(&file);
    let mut header = String::new();
    match reader.read_line(&mut header) {
        Ok(_) => {},
        Err(e) => panic!("Error while getting the header: {}",e),
    };
    let mut split_header = header.split_whitespace();

    let number_var = split_header.nth(2).unwrap().parse::<usize>().unwrap();
    let number_clauses = split_header.next().unwrap().parse::<usize>().unwrap();
    
    let mut g = Problem::new(state, number_var, number_clauses);
    
    let mut number_probabilistic = 0;
    for d in distributions.iter() {
        number_probabilistic += d.len();
    }
    let variable_mapping = g.add_distributions(&distributions, state);
    
    // Second pass to parse the clauses
    let file = File::open(filepath).unwrap();
    let reader = BufReader::new(file);
    for l in reader.lines() {
        match l {
            Err(e) => panic!("Problem while reading file: {}", e),
            Ok(line) => {
                if learn && line.starts_with("c p learn") {
                    for distribution in g.distributions_iter() {
                        g[distribution].set_branching_candidate(false);
                    }
                    for distribution in line.split_whitespace().skip(3).map(|s| DistributionIndex(s.parse::<usize>().unwrap() - 1)) {
                        g[distribution].set_branching_candidate(true);
                    }
                }
                if !line.starts_with('c') && !line.starts_with('p') {
                    // Note: the space before the 0 is important so that clauses like "1 -10 0" are correctly splitted
                    for clause in line.split(" 0").filter(|cl| !cl.is_empty()) {
                        let mut literals: Vec<Literal> = vec![];
                        let mut head: Option<Literal> = None;
                        for lit in clause.split_whitespace() {
                            let parsed_lit = lit.parse::<isize>().unwrap();
                            let is_positive = parsed_lit > 0;
                            let mut variable = parsed_lit.abs() as usize;
                            if let Some(new_var) = variable_mapping.get(&variable) {
                                variable = *new_var;
                            }
                            let trail_value_index = g[VariableIndex(variable - 1)].get_value_index();
                            let literal = Literal::from_variable(VariableIndex(variable - 1), is_positive, trail_value_index);
                            if literal.to_variable().0 < number_probabilistic {
                                // The variable is probabilistic, put at the end of the vector
                                literals.push(literal);
                            } else {
                                // The variable is deterministic, put at the beginning of the vector
                                literals.insert(0, literal);
                            }
                            if literal.is_positive() {
                                if head.is_some() {
                                    panic!("The clause {} has multiple positive literals", line);
                                }
                                head = Some(literal);
                            }
                        }
                        g.add_clause(literals, head, state, false);
                    }
                }
            }
        }
    }
    g
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

pub fn type_of_input(filepath: &PathBuf) -> FileType {
    let mut header = String::new();
    {
        let file = File::open(filepath).unwrap();
        let mut reader = BufReader::new(&file);
        match reader.read_line(&mut header) {
            Ok(_) => {},
            Err(e) => panic!("Error while getting the header: {}",e),
        };
    }
    if header.starts_with("p cnf") {
        FileType::CNF
    } else if header.starts_with("outputs") {
        FileType::FDAC
    } else {
        panic!("Unexpected file format to read from. Header does not match .cnf or .fdac file: {}", header);
    }
}
/*
#[cfg(test)]
mod test_ppidimacs_parsing {

    use super::problem_from_ppidimacs;
    use crate::core::problem::VariableIndex;
    use crate::core::trail::StateManager;
    use std::path::PathBuf;

    #[test]
    fn test_file() {
        let mut file = PathBuf::new();
        let mut state = StateManager::default();
        file.push("tests/instances/bayesian_networks/abc_chain_b0.ppidimacs");
        let (g, _) = problem_from_ppidimacs(&file, &mut state);
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
