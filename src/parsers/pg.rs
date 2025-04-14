use super::*;
use crate::core::problem::Problem;
use search_trail::StateManager;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use rustc_hash::FxHashMap;
use malachite::rational::Rational;
use crate::common::rational;

struct EdgeConstraint {
    source: isize,
    target: isize,
    parameter_variable: isize,
}

impl EdgeConstraint {
    fn new(source: isize, target: isize, parameter_variable: isize) -> Self {
        Self { source, target, parameter_variable }
    }

    fn to_cnf(&self, indicator_offset: isize) -> Vec<isize> {
        vec![-(self.source + indicator_offset), -self.parameter_variable, self.target + indicator_offset]
    }
}

pub struct PgParser {
    input: PathBuf,
    evidence: OsString,
}

impl PgParser {
    pub fn new(input: PathBuf, evidence: OsString) -> Self {
        Self { input, evidence }
    }
}

impl Parser for PgParser {

    fn problem_from_file(&self, state: &mut StateManager) -> Problem {
        let file = File::open(&self.input).unwrap();
        let reader = BufReader::new(&file);
        // Loading the content of the file. The file is loaded in a single String in which new line
        // have been removed. Then it is split by whitespace, giving only the numbers in the file.
        let content = reader.lines().map(|l| l.unwrap()).collect::<Vec<String>>().join(" ");
        let content = content.split_whitespace().collect::<Vec<&str>>();

        let directed = match content[0] {
            "DIRECTED" => true,
            "UNDIRECTED" => false,
            _ => panic!("Bad header for probalistic graph file: {}", content[0]),
        };
        let mut map_node_to_id: FxHashMap<&str, isize> = FxHashMap::default();
        let mut distributions: Vec<Vec<Rational>> = vec![];
        let mut clauses: Vec<EdgeConstraint> = vec![];
        let mut node_index = 1;
        let mut parameter_index = 1;
        let mut content_index = 1;
        while content_index < content.len() {
            let source = content[content_index];
            let target = content[content_index + 1];
            let proba_up = rational(content[content_index + 2].parse::<f64>().unwrap());
            content_index += 3;
            distributions.push(vec![proba_up.clone(), rational(1.0) - proba_up]);
            let source_id = if !map_node_to_id.contains_key(source) {
                map_node_to_id.insert(source, node_index);
                node_index += 1;
                node_index - 1
            } else {
                *map_node_to_id.get(source).unwrap()
            };
            let target_id = if !map_node_to_id.contains_key(target) {
                map_node_to_id.insert(target, node_index);
                node_index += 1;
                node_index - 1
            } else {
                *map_node_to_id.get(target).unwrap()
            };
            clauses.push(EdgeConstraint::new(source_id, target_id, parameter_index));
            if !directed {
                clauses.push(EdgeConstraint::new(target_id, source_id, parameter_index));
            }
            parameter_index += 2;
        }
        let mut clauses = clauses.iter().map(|c| c.to_cnf(parameter_index - 1)).collect::<Vec<Vec<isize>>>();

        let content = evidence_from_os_string(&self.evidence);
        let content = content.split_whitespace().collect::<Vec<&str>>();
        if content.len() != 2 {
            panic!("The evidence should be the source and target nodes (2 strings). Got: {}", content.join(" "));
        }
        let source = *map_node_to_id.get(&content[0]).unwrap_or_else(|| panic!("Source node {} is not in the graph structure", content[0]));
        let target = *map_node_to_id.get(&content[1]).unwrap_or_else(|| panic!("Target node {} is not in the graph structure", content[1]));
        clauses.push(vec![source + parameter_index - 1]);
        clauses.push(vec![-(target + parameter_index - 1)]);
        create_problem(&distributions, &clauses, state)
    }

    fn clauses_from_file(&self) -> Vec<Vec<isize>> {
        let file = File::open(&self.input).unwrap();
        let reader = BufReader::new(&file);
        // Loading the content of the file. The file is loaded in a single String in which new line
        // have been removed. Then it is split by whitespace, giving only the numbers in the file.
        let content = reader.lines().map(|l| l.unwrap()).collect::<Vec<String>>().join(" ");
        let content = content.split_whitespace().collect::<Vec<&str>>();

        let directed = match content[0] {
            "DIRECTED" => true,
            "UNDIRECTED" => false,
            _ => panic!("Bad header for probalistic graph file: {}", content[0]),
        };
        let mut map_node_to_id: FxHashMap<&str, isize> = FxHashMap::default();
        let mut distributions: Vec<Vec<Rational>> = vec![];
        let mut clauses: Vec<EdgeConstraint> = vec![];
        let mut node_index = 1;
        let mut parameter_index = 1;
        let mut content_index = 1;
        while content_index < content.len() {
            let source = content[content_index];
            let target = content[content_index + 1];
            let proba_up = content[content_index + 2].parse::<f64>().unwrap();
            content_index += 3;
            distributions.push(vec![rational(proba_up), rational(1.0 - proba_up)]);
            let source_id = if !map_node_to_id.contains_key(source) {
                map_node_to_id.insert(source, node_index);
                node_index += 1;
                node_index - 1
            } else {
                *map_node_to_id.get(source).unwrap()
            };
            let target_id = if !map_node_to_id.contains_key(target) {
                map_node_to_id.insert(target, node_index);
                node_index += 1;
                node_index - 1
            } else {
                *map_node_to_id.get(target).unwrap()
            };
            clauses.push(EdgeConstraint::new(source_id, target_id, parameter_index));
            if !directed {
                clauses.push(EdgeConstraint::new(target_id, source_id, parameter_index));
            }
            parameter_index += 2;
        }
        let mut clauses = clauses.iter().map(|c| c.to_cnf(parameter_index - 1)).collect::<Vec<Vec<isize>>>();

        let content = evidence_from_os_string(&self.evidence);
        let content = content.split_whitespace().collect::<Vec<&str>>();
        if content.len() != 2 {
            panic!("The evidence should be the source and target nodes (2 strings). Got: {}", content.join(" "));
        }
        let source = *map_node_to_id.get(&content[0]).unwrap_or_else(|| panic!("Source node {} is not in the graph structure", content[0]));
        let target = *map_node_to_id.get(&content[1]).unwrap_or_else(|| panic!("Target node {} is not in the graph structure", content[1]));
        clauses.push(vec![source + parameter_index - 1]);
        clauses.push(vec![-(target + parameter_index - 1)]);
        clauses
    }

    fn distributions_from_file(&self) -> Vec<Vec<Rational>> {
        vec![]
    }
}
