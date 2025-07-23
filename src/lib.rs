// Re-export the modules
mod solver;
mod logger;
pub mod args;
pub mod common;
mod branching;
pub mod core;
mod parsers;
mod propagator;
mod preprocess;
pub mod learner;
mod caching;
mod target;

use std::ffi::OsString;
use std::path::PathBuf;
use std::fs::File;
use std::io::{Write, BufRead, BufReader};
use malachite::rational::Rational;

use learner::Learner;
use search_trail::StateManager;

use core::components::ComponentExtractor;
use core::problem::Problem;
use parsers::*;

use propagator::Propagator;
pub use common::*;
use branching::*;
use caching::*;
use args::*;

pub use solver::Solver;
use solver::SolverParameters;

use peak_alloc::PeakAlloc;
#[global_allocator]
pub static PEAK_ALLOC: PeakAlloc = PeakAlloc;

pub fn solve(args: Args) -> f64 {
    let mut state = StateManager::default();
    let propagator = Propagator::new(&mut state);
    let parser = parser_from_input(args.input().clone(), args.query().clone());
    let problem = parser.problem_from_file(&mut state);
    let caching_scheme = CachingScheme::new(args.caching());
    let component_extractor = ComponentExtractor::new(&problem, caching_scheme, &mut state);
    let mut solver = generic_solver(problem, state, component_extractor, propagator, &args);
    let parameters = SolverParameters::new(&args);
    let solution = solver.solve(&parameters);
    if !args.statistics() {
        solution.print();
    }
    solution.to_f64()
}

pub fn parse_csv(filename: PathBuf) -> Vec<(OsString, f64)> {
    let mut ret: Vec<(OsString, f64)> = vec![];
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    for line in reader.lines().skip(1) {
        let l = line.unwrap();
        let split = l.split(',').collect::<Vec<&str>>();
        ret.push((split[0].parse::<OsString>().unwrap(), split[1].parse::<f64>().unwrap()));
    }
    ret
}

pub fn learn(args: Args) {
    let input = args.input().clone();
    if args.statistics() {
        let mut learner = Learner::<true>::new(input, &args);
        learner.train(&args);
    } else {
        let mut learner = Learner::<false>::new(input, &args);
        learner.train(&args);
    };
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
    Log(Solver<true>),
    NoLog(Solver<false>),
}

impl GenericSolver {
    fn solve(&mut self, parameters: &SolverParameters) -> Solution {
        match self {
            Self::Log(solver) => solver.compute_pwmc(parameters),
            Self::NoLog(solver) => solver.compute_pwmc(parameters),
        }
    }
}

pub fn generic_solver(problem: Problem, state: StateManager, component_extractor: ComponentExtractor, propagator: Propagator, args: &Args) -> GenericSolver {
    let branching: Box<dyn BranchingDecision> = match args.branching() {
        Branching::MinInDegree => Box::<MinInDegree>::default(),
        Branching::MinOutDegree => Box::<MinOutDegree>::default(),
        Branching::DLCS => Box::<DLCS>::default(),
        Branching::DLCSVar => Box::<DLCSVar>::default(),
    };
    if args.statistics() {
        let solver = Solver::<true>::new(problem, state, component_extractor, branching, propagator);
        GenericSolver::Log(solver)
    } else {
        let solver = Solver::<false>::new(problem, state, component_extractor, branching, propagator);
        GenericSolver::NoLog(solver)
    }
}
