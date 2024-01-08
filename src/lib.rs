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

use learning::learner::Learner;
use learning::Learning;
use learning::tensor_learner::TensorLearner;
use diagrams::dac::dac::Dac;
use solvers::compiler::DACCompiler;
use sysinfo::{SystemExt, System};
use search_trail::StateManager;
use clap::ValueEnum;
use rug::Float;

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
pub mod learning;
pub mod diagrams;

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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Loss {
    MAE,
    MSE,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Semiring {
    Probability,
    Tensor,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Optimizer {
    Adam,
    SGD,
}

pub fn compile(input: PathBuf, branching: Branching, fdac: Option<PathBuf>, dotfile: Option<PathBuf>) -> Option<Dac<Float>>{
    match type_of_input(&input) {
        FileType::CNF => {
            let mut compiler = make_compiler!(&input, branching, 0.0);
            let mut res = compile!(compiler);
            if let Some(ref mut dac) = &mut res {
                dac.optimize_structure();
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
            res
        },
        FileType::FDAC => {
            let mut dac = Dac::from_file(&input);
            dac.evaluate();
            println!("{}", dac.get_circuit_probability());
            Some(dac)
        },
    }
}

pub fn make_learner(inputs: Vec<PathBuf>, expected: Vec<f64>, epsilon: f64, branching: Branching, outfolder: Option<PathBuf>, jobs: usize, log: bool, semiring: Semiring, optimizer: Optimizer) -> Box<dyn Learning> {
    match semiring {
        Semiring::Probability => {
            if log {
                Box::new(Learner::<true>::new(inputs, expected, epsilon, branching, outfolder, jobs))
            } else {
                Box::new(Learner::<false>::new(inputs, expected, epsilon, branching, outfolder, jobs))
            }
        },
        Semiring::Tensor => {
            if log {
                Box::new(TensorLearner::<true>::new(inputs, expected, epsilon, branching, outfolder, jobs, optimizer))
            } else {
                Box::new(TensorLearner::<false>::new(inputs, expected, epsilon, branching, outfolder, jobs, optimizer))
            }
        }
    }
}

pub fn learn(trainfile: PathBuf, branching: Branching, outfolder: Option<PathBuf>, lr:f64, nepochs: usize, 
            log:bool, timeout:i64, epsilon: f64, loss: Loss, jobs: usize, semiring: Semiring, optimizer: Optimizer) {    
    // Sets the number of threads for rayon
    let mut inputs = vec![];
    let mut expected: Vec<f64> = vec![];
    let file = File::open(&trainfile).unwrap();
    let reader = BufReader::new(file);
    for line in reader.lines().skip(1) {
        let l = line.unwrap();
        let mut split = l.split(",");
        inputs.push(split.next().unwrap().parse::<PathBuf>().unwrap());
        expected.push(split.next().unwrap().parse::<f64>().unwrap());
    }
    let mut learner = make_learner(inputs, expected, epsilon, branching, outfolder, jobs, log, semiring, optimizer);
    learner.train(nepochs, lr, loss, timeout);
}

pub fn search(input: PathBuf, branching: Branching, statistics: bool, memory: Option<u64>, epsilon: f64) -> ProblemSolution {
    let solver = make_solver!(&input, branching, epsilon, memory, statistics);
    solve_search!(solver)
}


impl std::fmt::Display for Loss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Loss::MAE => write!(f, "MAE (Mean Absolute Error)"),
            Loss::MSE => write!(f, "MSE (Mean Squared Error)"),
        }
    }
}
