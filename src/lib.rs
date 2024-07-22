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

use std::path::PathBuf;
use std::fs::File;
use std::io::{Write, BufRead, BufReader};
use rug::Float;

use learning::{Learning, learner::Learner, LearnParameters};
use ac::ac::Dac;
use search_trail::StateManager;

use core::components::ComponentExtractor;
use common::Solution;
use core::problem::Problem;
use parser::*;

use propagator::Propagator;
use common::*;
use branching::*;

pub use solver::Solver;
use solver::SolverParameters;

// Re-export the modules
mod solver;
mod statistics;
pub mod common;
mod branching;
pub mod core;
mod parser;
mod propagator;
mod preprocess;
pub mod learning;
pub mod ac;
mod semiring;

use peak_alloc::PeakAlloc;
#[global_allocator]
pub static PEAK_ALLOC: PeakAlloc = PeakAlloc;

pub fn solve_from_problem(distributions: &[Vec<f64>], clauses: &[Vec<isize>], branching: Branching, epsilon: f64, memory: Option<u64>, timeout: u64, statistics: bool) -> Solution {
    let parameters = SolverParameters::new(if let Some(m) = memory { m } else { u64::MAX }, epsilon, timeout);
    let solver = solver_from_problem!(distributions, clauses, branching, parameters, statistics);
    match solver {
        GenericSolver::SMinInDegree(mut solver) => solver.search(false),
        GenericSolver::QMinInDegree(mut solver) => solver.search(false),
    }
}

pub fn search(input: PathBuf, branching: Branching, statistics: bool, memory: Option<u64>, epsilon: f64, approx: ApproximateMethod, timeout: u64) -> f64 {
    let parameters = SolverParameters::new(if let Some(m) = memory { m } else { u64::MAX}, epsilon, timeout);
    let solver = make_solver!(&input, branching, parameters, statistics);
    let solution = match approx {
        ApproximateMethod::Bounds => {
            match solver {
                GenericSolver::SMinInDegree(mut solver) => solver.search(false),
                GenericSolver::QMinInDegree(mut solver) => solver.search(false),
            }
        },
        ApproximateMethod::LDS => {
            match solver {
                GenericSolver::SMinInDegree(mut solver) => solver.search(true),
                GenericSolver::QMinInDegree(mut solver) => solver.search(true),
            }
        },
    };
    solution.print();
    solution.to_f64()
}

fn _compile(compiler: GenericSolver, approx: ApproximateMethod, fdac: Option<PathBuf>, dotfile: Option<PathBuf>) -> Solution {
    let mut dac: Dac<Float> = match approx {
        ApproximateMethod::Bounds => {
            match compiler {
                GenericSolver::SMinInDegree(mut solver) => solver.compile(false),
                GenericSolver::QMinInDegree(mut solver) => solver.compile(false),
            }
        },
        ApproximateMethod::LDS => {
            match compiler {
                GenericSolver::SMinInDegree(mut solver) => solver.compile(true),
                GenericSolver::QMinInDegree(mut solver) => solver.compile(true),
            }
        },
    };
    dac.evaluate();
    if let Some(f) = dotfile {
        let out = dac.as_graphviz();
        let mut outfile = File::create(f).unwrap();
        match outfile.write_all(out.as_bytes()) {
            Ok(_) => (),
            Err(e) => println!("Could not write the circuit into the dot file: {:?}", e),
        }
    }
    if let Some(f) = fdac {
        let mut outfile = File::create(f).unwrap();
        match outfile.write_all(format!("{}", dac).as_bytes()) {
            Ok(_) => (),
            Err(e) => println!("Could not write the circuit into the fdac file: {:?}", e),
        }
        
    }
    dac.solution()
}

pub fn compile(input: PathBuf, branching: Branching, fdac: Option<PathBuf>, dotfile: Option<PathBuf>, epsilon: f64, approx: ApproximateMethod, timeout: u64) -> f64 {
    let parameters = SolverParameters::new(u64::MAX, epsilon, timeout);
    let solution = match type_of_input(&input) {
        FileType::CNF => {
            let compiler = make_solver!(&input, branching, parameters, false);
            _compile(compiler, approx, fdac, dotfile)
        },
        FileType::FDAC => {
            let mut dac: Dac<Float> = Dac::<Float>::from_file(&input);
            dac.evaluate();
            dac.solution()
        },
    };
    solution.print();
    solution.to_f64()
}

pub fn compile_from_problem(distributions: &[Vec<f64>], clauses: &[Vec<isize>], branching: Branching, epsilon: f64, approx: ApproximateMethod, timeout: u64, statistics: bool, fdac: Option<PathBuf>, dotfile: Option<PathBuf>) -> Solution {
    let parameters = SolverParameters::new(u64::MAX, epsilon, timeout);
    let solver = solver_from_problem!(distributions, clauses, branching, parameters, statistics);
    _compile(solver, approx, fdac, dotfile)
}


pub fn make_learner(inputs: Vec<PathBuf>, expected: Vec<f64>, epsilon: f64, approx:ApproximateMethod, branching: Branching, outfolder: Option<PathBuf>, jobs: usize, log: bool, semiring: Semiring, params: &LearnParameters, test_inputs:Vec<PathBuf>, test_expected:Vec<f64>) -> Box<dyn Learning> {
    match semiring {
        Semiring::Probability => {
            if log {
                Box::new(Learner::<true>::new(inputs, expected, epsilon, approx, branching, outfolder, jobs, params.compilation_timeout(), test_inputs, test_expected))
            } else {
                Box::new(Learner::<false>::new(inputs, expected, epsilon, approx, branching, outfolder, jobs, params.compilation_timeout(), test_inputs, test_expected))
            }
        },
    }
}

pub fn learn(trainfile: PathBuf, testfile:Option<PathBuf>, branching: Branching, outfolder: Option<PathBuf>, 
            log:bool, epsilon: f64, approx: ApproximateMethod, jobs: usize, semiring: Semiring, params: LearnParameters) {    
    // Sets the number of threads for rayon
    let mut inputs = vec![];
    let mut expected: Vec<f64> = vec![];
    let file = File::open(trainfile).unwrap();
    let reader = BufReader::new(file);
    for line in reader.lines().skip(1) {
        let l = line.unwrap();
        let mut split = l.split(',');
        inputs.push(split.next().unwrap().parse::<PathBuf>().unwrap());
        expected.push(split.next().unwrap().parse::<f64>().unwrap());
    }
    let mut test_inputs = vec![];
    let mut test_expected: Vec<f64> = vec![];
    if let Some(testfile) = testfile {
        let file = File::open(testfile).unwrap();
        let reader = BufReader::new(file);
        for line in reader.lines().skip(1) {
            let l = line.unwrap();
            let mut split = l.split(',');
            test_inputs.push(split.next().unwrap().parse::<PathBuf>().unwrap());
            test_expected.push(split.next().unwrap().parse::<f64>().unwrap());
        }
    }
    let mut learner = make_learner(inputs, expected, epsilon, approx, branching, outfolder, jobs, log, semiring, &params, test_inputs, test_expected);
    learner.train(&params);
}

impl std::fmt::Display for Loss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Loss::MAE => write!(f, "MAE (Mean Absolute Error)"),
            Loss::MSE => write!(f, "MSE (Mean Squared Error)"),
        }
    }
}

pub enum GenericSolver {
    SMinInDegree(Solver<MinInDegree, true>),
    QMinInDegree(Solver<MinInDegree, false>),
}

pub fn generic_solver(problem: Problem, state: StateManager, component_extractor: ComponentExtractor, branching: Branching, propagator: Propagator, parameters: SolverParameters, stat: bool) -> GenericSolver {
    if stat {
        match branching {
            Branching::MinInDegree => {
                let solver = Solver::<MinInDegree, true>::new(problem, state, component_extractor, Box::<MinInDegree>::default(), propagator, parameters);
                GenericSolver::SMinInDegree(solver)
            },
        }
    } else {
        match branching {
            Branching::MinInDegree => {
                let solver = Solver::<MinInDegree, false>::new(problem, state, component_extractor, Box::<MinInDegree>::default(), propagator, parameters);
                GenericSolver::QMinInDegree(solver)
            },
        }
    }
}

macro_rules! solver_from_problem {
    ($d:expr, $c:expr, $b:expr, $p:expr, $s:expr) => {
        {
            let mut state = StateManager::default();
            let problem = problem_from_problem($d, $c, &mut state);
            let propagator = Propagator::new(&mut state);
            let component_extractor = ComponentExtractor::new(&problem, &mut state);
            generic_solver(problem, state, component_extractor, $b, propagator, $p, $s)
        }
    };
}
use solver_from_problem;

macro_rules! make_solver {
    ($i:expr, $b:expr, $p:expr, $s:expr) => {
        {
            let mut state = StateManager::default();
            let propagator = Propagator::new(&mut state);
            let problem = problem_from_cnf($i, &mut state, false);
            let component_extractor = ComponentExtractor::new(&problem, &mut state);
            generic_solver(problem, state, component_extractor, $b, propagator, $p, $s)
        }
    };
}
pub(crate) use make_solver;
