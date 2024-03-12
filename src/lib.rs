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

use learning::{learner::Learner, LearnParameters};
use learning::Learning;
#[cfg(feature = "tensor")]
use learning::tensor_learner::TensorLearner;
use diagrams::dac::dac::Dac;
use solvers::GenericSolver;
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
pub mod solvers;
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
    #[cfg(feature = "tensor")]
    Tensor,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Optimizer {
    Adam,
    SGD,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum ApproximateMethod {
    /// Bound-based pruning
    Bounds,
    /// Limited Discrepancy Search
    LDS,
}

impl std::fmt::Display for ApproximateMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApproximateMethod::Bounds => write!(f, "bounds"),
            ApproximateMethod::LDS => write!(f, "lds"),
        }
    }
}

pub fn compile(input: PathBuf, branching: Branching, fdac: Option<PathBuf>, dotfile: Option<PathBuf>, epsilon: f64, timeout: u64) -> ProblemSolution {
    match type_of_input(&input) {
        FileType::CNF => {
            let compiler = make_solver!(&input, branching, epsilon, None, timeout, false);
            let mut res: Option<Dac<Float>> = compile!(compiler);
            if let Some(ref mut dac) = &mut res {
                dac.evaluate();
                let proba = dac.circuit_probability().clone();
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
                ProblemSolution::Ok((proba.clone(), proba.clone()))
            } else {
                ProblemSolution::Err(Error::Timeout)
            }
        },
        FileType::FDAC => {
            let mut dac: Dac<Float> = Dac::<Float>::from_file(&input);
            dac.evaluate();
            let proba = dac.circuit_probability().clone();
            ProblemSolution::Ok((proba.clone(), proba.clone()))
        },
    }
}

pub fn make_learner(inputs: Vec<PathBuf>, expected: Vec<f64>, epsilon: f64, branching: Branching, outfolder: Option<PathBuf>, jobs: usize, log: bool, semiring: Semiring, params: &LearnParameters, test_inputs:Vec<PathBuf>, test_expected:Vec<f64>) -> Box<dyn Learning> {
    match semiring {
        Semiring::Probability => {
            if log {
                Box::new(Learner::<true>::new(inputs, expected, epsilon, branching, outfolder, jobs, params.timeout(), test_inputs, test_expected))
            } else {
                Box::new(Learner::<false>::new(inputs, expected, epsilon, branching, outfolder, jobs, params.timeout(), test_inputs, test_expected))
            }
        },
        #[cfg(feature = "tensor")]
        Semiring::Tensor => {
            if log {
                Box::new(TensorLearner::<true>::new(inputs, expected, epsilon, branching, outfolder, jobs, params.timeout(), params.optimizer(), test_inputs, test_expected))
            } else {
                Box::new(TensorLearner::<false>::new(inputs, expected, epsilon, branching, outfolder, jobs, params.timeout(), params.optimizer(), test_inputs, test_expected))
            }
        }
    }
}

pub fn learn(trainfile: PathBuf, testfile:Option<PathBuf>, branching: Branching, outfolder: Option<PathBuf>, 
            log:bool, epsilon: f64, jobs: usize, semiring: Semiring, params: LearnParameters) {    
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
    let mut test_inputs = vec![];
    let mut test_expected: Vec<f64> = vec![];
    if let Some(testfile) = testfile {
        let file = File::open(&testfile).unwrap();
        let reader = BufReader::new(file);
        for line in reader.lines().skip(1) {
            let l = line.unwrap();
            let mut split = l.split(",");
            test_inputs.push(split.next().unwrap().parse::<PathBuf>().unwrap());
            test_expected.push(split.next().unwrap().parse::<f64>().unwrap());
        }
    }
    let mut learner = make_learner(inputs, expected, epsilon, branching, outfolder, jobs, log, semiring, &params, test_inputs, test_expected);
    learner.train(&params);
}

pub fn search(input: PathBuf, branching: Branching, statistics: bool, memory: Option<u64>, epsilon: f64, approx: ApproximateMethod, timeout: u64, discrepancy_threshold: f64) -> ProblemSolution {
    let solver = make_solver!(&input, branching, epsilon, memory, timeout, statistics);
    match approx {
        ApproximateMethod::Bounds => search!(solver),
        ApproximateMethod::LDS => {
            lds!(solver, discrepancy_threshold)
        },
    }
}


impl std::fmt::Display for Loss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Loss::MAE => write!(f, "MAE (Mean Absolute Error)"),
            Loss::MSE => write!(f, "MSE (Mean Squared Error)"),
        }
    }
}
