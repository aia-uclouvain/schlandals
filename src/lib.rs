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
use std::{path::PathBuf, fs::File, io::{Write,BufRead,BufReader}};

use learning::{LogLearner, QuietLearner, circuit::Dac};
use learning::exact::DACCompiler;
use sysinfo::{SystemExt, System};
use search_trail::StateManager;
use clap::ValueEnum;

use crate::core::components::ComponentExtractor;
use crate::branching::*;
use solvers::ProblemSolution;
use crate::solvers::*;
use crate::parser::*;

use propagator::Propagator;

// Re-export the modules
mod common;
mod branching;
pub mod core;
mod solvers;
mod parser;
mod propagator;
mod preprocess;
mod learning;

use peak_alloc::PeakAlloc;
#[global_allocator]
pub static PEAK_ALLOC: PeakAlloc = PeakAlloc;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Branching {
    /// Minimum In-degree of a clause in the implication-graph
    MinInDegree,
    /// Minimum Out-degree of a clause in the implication-graph
    MinOutDegree,
    /// Maximum degree of a clause in the implication-graph
    MaxDegree,
    /// Variable State Independent Decaying Sum
    VSIDS,
}

pub fn compile(input: PathBuf, branching: Branching, ratio: f64, fdac: Option<PathBuf>, dotfile: Option<PathBuf>) {
    match type_of_input(&input) {
        FileType::CNF => {
            let number_distribution = distributions_from_cnf(&input).len();
            let limit = (ratio * number_distribution as f64).floor() as usize;
            let compiler = make_compiler!(&input, branching, limit);
            if let Some(mut dac) = compile!(compiler, u64::MAX) {
                dac.evaluate();
                if let Some(f) = dotfile {
                    let out = dac.as_graphviz();
                    let mut outfile = File::create(f).unwrap();
                    match outfile.write(out.as_bytes()) {
                        Ok(_) => (),
                        Err(e) => println!("Could not write the circuit into the dot file: {:?}", e),
                    }
                }
                if let Some(f) = fdac {
                    let mut outfile = File::create(f).unwrap();
                    match outfile.write(format!("{}", dac).as_bytes()) {
                        Ok(_) => (),
                        Err(e) => println!("Could not write the circuit into the fdac file: {:?}", e),
                    }
                    
                }
            }
        },
        FileType::FDAC => {
            let mut dac = Dac::from_file(&input);
            dac.evaluate();
            println!("{}", dac.get_circuit_probability());
        },
    }
}

pub fn learn(trainfile: PathBuf, branching: Branching, outfolder: Option<PathBuf>, lr:f64, nepochs: usize, 
            log:bool, timeout:u64, rlearned: Option<f64>, epsilon: f64) {    
    let mut inputs = vec![];
    let mut expected: Vec<f64> = vec![];
    let file = File::open(&trainfile).unwrap();
    let reader = BufReader::new(file);
    for line in reader.lines().skip(1) {
        let l = line.unwrap();
        let mut split = l.split(",");
        inputs.push(trainfile.parent().unwrap().join(split.next().unwrap().parse::<PathBuf>().unwrap()));
        expected.push(split.next().unwrap().parse::<f64>().unwrap());
    }
    let ratio_learned = match rlearned {
        None => 1.0,
        Some(r) => r,
    };
    if log { 
        let mut learner = LogLearner::new(inputs, expected, epsilon, branching, timeout, outfolder, ratio_learned);
        learner.train(nepochs, lr);
    } else {
        let mut learner = QuietLearner::new(inputs, expected, epsilon, branching, timeout, outfolder, ratio_learned);
        learner.train(nepochs, lr);
    }
}

pub fn search(input: PathBuf, branching: Branching, statistics: bool, memory: Option<u64>, epsilon: f64) -> ProblemSolution {
    let solver = make_solver!(&input, branching, epsilon, memory, statistics);
    solve_search!(solver)
}
