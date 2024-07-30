//Schlandals
//Copyright (C) 2022-2023 A. Dubray, L. Dierckx
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

use super::*;
use crate::core::problem::Problem;
use search_trail::StateManager;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use rustc_hash::FxHashMap;

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
        let mut distributions: Vec<Vec<f64>> = vec![];
        let mut clauses: Vec<EdgeConstraint> = vec![];
        let mut node_index = 1;
        let mut parameter_index = 1;
        let mut content_index = 1;
        while content_index < content.len() {
            let source = content[content_index];
            let target = content[content_index + 1];
            let proba_up = content[content_index + 2].parse::<f64>().unwrap();
            content_index += 3;
            distributions.push(vec![proba_up, 1.0 - proba_up]);
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
        let number_evidence = content[0].parse::<usize>().unwrap();
        let mut content_index = 1;
        for _ in 0..number_evidence {
            let node = *map_node_to_id.get(&content[content_index]).unwrap_or_else(|| panic!("The node {} is in the evidence file but not in the problem file", content[content_index]));
            let value = content[content_index + 1].parse::<usize>().unwrap();
            if value == 0 {
                clauses.push(vec![-(node + parameter_index - 1)]);
            } else {
                clauses.push(vec![node + parameter_index - 1]);
            }
            content_index += 2
        }
        create_problem(&distributions, &clauses, state)
    }

    fn distributions_from_file(&self) -> Vec<Vec<f64>> {
        vec![]
    }
}
