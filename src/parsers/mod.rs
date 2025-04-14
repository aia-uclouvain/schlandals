pub mod cnf;
pub mod uai;
pub mod pg;

use crate::core::problem::{Problem, VariableIndex};
use crate::core::literal::Literal;
use search_trail::StateManager;
use std::ffi::OsString;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::io::{BufRead, BufReader};
use malachite::rational::Rational;

use cnf::*;
use uai::*;
use pg::*;

pub trait Parser {
    fn problem_from_file(&self, state: &mut StateManager) -> Problem;
    fn distributions_from_file(&self) -> Vec<Vec<Rational>>;
    fn clauses_from_file(&self) -> Vec<Vec<isize>>;
}

pub fn parser_from_input(filepath: PathBuf, evidence: Option<OsString>) -> Box<dyn Parser + 'static > {
    let mut header = String::new();
    {
        let file = File::open(&filepath).unwrap();
        let mut reader = BufReader::new(&file);
        match reader.read_line(&mut header) {
            Ok(_) => {},
            Err(e) => panic!("Error while getting the header: {}",e),
        };
    }
    if header.starts_with("p cnf") {
        Box::new(CnfParser::new(filepath, evidence.unwrap_or(OsString::default())))
    } else if header.starts_with("BAYES") {
        Box::new(UaiParser::new(filepath, evidence.unwrap()))
    } else if header.starts_with("DIRECTED") || header.starts_with("UNDIRECTED") {
        Box::new(PgParser::new(filepath, evidence.unwrap()))
    } else {
        panic!("Unexpected file format to read from. Header does not match .cnf or .fdac file: {}", header);
    }
}

pub fn evidence_from_os_string(evidence: &OsString) -> String {
    match Path::new(evidence).try_exists() {
        Ok(v) => {
            if v {
                let file = File::open(evidence).unwrap();
                let reader = BufReader::new(&file);
                reader.lines().map(|l| l.unwrap()).collect::<Vec<String>>().join(" ")
            } else {
                evidence.to_str().unwrap().to_string()
            }
        },
        Err(e) => panic!("Error while checking if evidence file exists or not: {}", e),
    }
}

pub fn create_problem(distributions: &[Vec<Rational>], clauses: &[Vec<isize>], state: &mut StateManager) -> Problem {
    let mut number_var = 0;
    for clause in clauses.iter() {
        number_var = number_var.max(clause.iter().map(|l| l.unsigned_abs()).max().unwrap());
    }
    let mut problem = Problem::new(state, number_var, clauses.len());
    let variable_mapping = problem.add_distributions(distributions, state);
    for clause in clauses.iter() {
        let mut literals: Vec<Literal> = vec![];
        let mut head: Option<Literal> = None;
        for lit in clause.iter().copied() {
            if lit == 0 {
                panic!("Variables in clauses can not be 0");
            }
            let mut variable = lit.unsigned_abs();
            if let Some(new_variable) = variable_mapping.get(&variable) {
                variable = *new_variable;
            }
            let var = VariableIndex(variable - 1);
            let trail_value_index = problem[var].get_value_index();
            let literal = Literal::from_variable(var, lit > 0, trail_value_index);
            if lit > 0 {
                if head.is_some() {
                    panic!("The clauses {} has more than one positive literal", clause.iter().map(|i| format!("{}", i)).collect::<Vec<String>>().join(" "));
                }
                head = Some(literal);
            }
            literals.push(literal);
        }
        problem.add_clause(literals, head, state, false);
    }
    problem
}
