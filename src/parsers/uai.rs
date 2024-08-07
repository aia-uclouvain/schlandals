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

struct CPTConstraint {
    indicator_variables: Vec<usize>,
    parameter_variable: Option<isize>,
}

impl CPTConstraint {
    fn new(indicator_variables: Vec<usize>, parameter_variable: Option<isize>) -> Self {
        Self { indicator_variables, parameter_variable}
    }

    fn to_cnf(&self, indicator_offset: isize) -> Vec<isize> {
        let mut clause: Vec<isize> = vec![];
        if let Some(v) = self.parameter_variable {
            clause.push(-v);
        }
        let head = *self.indicator_variables.last().unwrap() as isize;
        for indicator_variable in self.indicator_variables.iter().copied().map(|c| c as isize) {
            if indicator_variable != head {
                clause.push(-(indicator_variable + indicator_offset));
            } else {
                clause.push(indicator_variable + indicator_offset);
            }
        }
        clause
    }
}

pub struct UaiParser {
    input: PathBuf,
    evidence: OsString,
}

impl UaiParser {

    pub fn new(input: PathBuf, evidence: OsString) -> Self {
        Self { input, evidence }
    }

    fn parent_values_from_domain(&self, domains: Vec<usize>) -> Vec<Vec<usize>> {
        let mut values: Vec<Vec<usize>> = vec![];
        let mut value = vec![0; domains.len()];
        let nb_values = domains.iter().product::<usize>();
        for _ in 0..nb_values {
            values.push(value.clone());
            for i in (0..value.len()).rev() {
                value[i] = (value[i] + 1) % domains[i];
                if value[i] != 0 {
                    break;
                }
            }
        }
        values
    }

}

impl Parser for UaiParser {

    fn problem_from_file(&self, state: &mut StateManager) -> Problem {
        let file = File::open(&self.input).unwrap();
        let reader = BufReader::new(&file);
        // Loading the content of the file. The file is loaded in a single String in which new line
        // have been removed. Then it is split by whitespace, giving only the numbers in the file.
        let content = reader.lines().skip(1).map(|l| l.unwrap()).collect::<Vec<String>>().join(" ");
        let content = content.split_whitespace().collect::<Vec<&str>>();
        // Parsing the preamble of the file containing the number of variables, their domain size, the
        // number of factor (must be equal to the number of variables), and their scope.
        let number_var = content[0].parse::<usize>().unwrap();
        let variables_domain_size = (0..number_var).map(|i| content[1 + i].parse::<usize>().unwrap()).collect::<Vec<usize>>();
        let mut variables_indicators: Vec<Vec<usize>> = vec![];
        let mut indicator = 1;
        for dom_size in variables_domain_size.iter().copied() {
            let indicators = (0..dom_size).map(|i| indicator + i).collect::<Vec<usize>>();
            indicator += indicators.len();
            variables_indicators.push(indicators);
        }
        let number_cpt = content[number_var + 1].parse::<usize>().unwrap();
        if number_cpt != number_var {
            panic!("The file declares {} variables but {} CPTs. In Bayesian network there must be one CPT per varaible", number_var, number_cpt);
        }
        let mut cpts_scope = (0..number_cpt).map(|_| vec![]).collect::<Vec<Vec<usize>>>();
        let mut content_index = number_var + 2;
        for cpt_scope in cpts_scope.iter_mut() {
            let scope_size = content[content_index].parse::<usize>().unwrap();
            for j in 0..scope_size {
                cpt_scope.push(content[content_index + 1 + j].parse::<usize>().unwrap());
            }
            content_index += scope_size + 1;
        }

        let mut distributions: Vec<Vec<f64>> = vec![];
        let mut clauses: Vec<CPTConstraint> = vec![];

        // Now we parse the actual factors with their probabilities.
        let mut variable_index = 1;
        for (cpt_index, cpt_scope) in cpts_scope.iter().enumerate() {
            let number_entry = cpt_scope.iter().copied().map(|variable| variables_domain_size[variable]).product::<usize>();
            if number_entry != content[content_index].parse::<usize>().unwrap() {
                panic!("Error while loading CPT for factor {}; {} entries are declared in the file, but the cartesian product of the variables in the scope is of size {}", cpt_index, content[content_index].parse::<usize>().unwrap(), number_entry);
            }
            content_index += 1;

            // Cache used to detect if multiple line in the CPT represent the same distribution
            let mut distribution_cache: FxHashMap<String, Vec<isize>> = FxHashMap::default();
            // The domain size of the variable in the CPT
            let scope_variables_domain = cpt_scope.iter().map(|v| variables_domain_size[*v]).collect::<Vec<usize>>();
            // The size of the distributions is the domain size of the last variable in the scope of
            // the factor
            let distribution_size = variables_domain_size[*scope_variables_domain.last().unwrap()];
            // How many distribution there are in the CPT
            let number_distribution = scope_variables_domain.iter().rev().skip(1).product::<usize>();
            // For each entry in the CPT, these are the value of the variables in the scope of the
            // factor associated with the entry.
            let variable_choices = self.parent_values_from_domain(scope_variables_domain);

            let mut choice_idx = 0;
            for _ in 0..number_distribution {
                let distribution = (0..distribution_size).map(|j| content[content_index + j].parse::<f64>().unwrap()).collect::<Vec<f64>>();
                content_index += distribution_size;

                let distribution_no_zero = distribution.iter().copied().filter(|p| *p != 0.0).collect::<Vec<f64>>();
                if distribution_no_zero.len() == 1 {
                    for (i, p) in distribution.iter().copied().enumerate() {
                        if p != 0.0 {
                            let indicator_variables = cpt_scope.iter().zip(variable_choices[choice_idx + i].iter()).map(|(variable, domain_idx)| variables_indicators[*variable][*domain_idx]).collect::<Vec<usize>>();
                            clauses.push(CPTConstraint::new(indicator_variables, None));
                            break;
                        }
                    }
                    choice_idx += distribution_size;
                    continue;
                }
                let mut cache_key = distribution_no_zero.clone();
                cache_key.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let cache_key = cache_key.iter().map(|p| format!("{}", p)).collect::<Vec<String>>().join(" ");
                match distribution_cache.get(&cache_key) {
                    Some(v) => {
                        let mut sorted_proba = distribution.iter().copied().enumerate().collect::<Vec<(usize, f64)>>();
                        sorted_proba.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        let mut v_idx = 0;
                        for (j, p) in sorted_proba.iter().copied() {
                            if p == 0.0 {
                                continue;
                            }
                            let indicator_variables = cpt_scope.iter().zip(variable_choices[choice_idx + j].iter()).map(|(variable, domain_idx)| variables_indicators[*variable][*domain_idx]).collect::<Vec<usize>>();
                            clauses.push(CPTConstraint::new(indicator_variables, Some(v[v_idx])));
                            v_idx += 1;
                        }
                        choice_idx += distribution_size;
                    },
                    None => {
                        distributions.push(distribution_no_zero);
                        let mut cache_entry: Vec<(f64, isize)> = vec![];
                        for probability in distribution.iter().copied() {
                            if probability == 0.0 {
                                choice_idx += 1;
                                continue;
                            }
                            let indicator_variables = cpt_scope.iter().zip(variable_choices[choice_idx].iter()).map(|(variable, domain_idx)| variables_indicators[*variable][*domain_idx]).collect::<Vec<usize>>();
                            choice_idx += 1;
                            clauses.push(CPTConstraint::new(indicator_variables,Some(variable_index)));
                            cache_entry.push((probability, variable_index));
                            variable_index += 1;
                        }
                        cache_entry.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        let cache_entry = cache_entry.iter().map(|x| x.1).collect::<Vec<isize>>();
                        distribution_cache.insert(cache_key, cache_entry);

                    }
                };
            }
        }
        let mut clauses = clauses.iter().map(|c| c.to_cnf(variable_index - 1)).collect::<Vec<Vec<isize>>>();

        let content = evidence_from_os_string(&self.evidence);
        let content = content.split_whitespace().map(|x| x.parse::<usize>().unwrap()).collect::<Vec<usize>>();
        let number_evidence = content[0];
        let mut content_index = 1;
        for _ in 0..number_evidence {
            let variable = content[content_index];
            let value = content[content_index + 1];
            for v in variables_indicators[variable].iter().copied().enumerate().filter(|(i, _)| *i != value).map(|(_, v)| -(v as isize + variable_index - 1)) {
                clauses.push(vec![v]);
            }
            content_index += 2
        }
        create_problem(&distributions, &clauses, state)
    }

    fn distributions_from_file(&self) -> Vec<Vec<f64>> {
        vec![]
    }
}
