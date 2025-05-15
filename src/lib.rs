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
pub mod ac;
mod caching;

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
    let solver = generic_solver(problem, state, component_extractor, propagator, &args);
    let parameters = SolverParameters::new(&args);
    let solution = if !args.compile() {
        match solver {
            GenericSolver::Search(mut solver) => solver.search(&parameters),
            GenericSolver::LogSearch(mut solver) => solver.search(&parameters),
            _ => panic!("Non-search solver used in search"),
        }
    } else {
        let mut ac = match solver {
            GenericSolver::Compiler(mut solver) => solver.compile(&parameters),
            GenericSolver::LogCompiler(mut solver) => solver.compile(&parameters),
            _ => panic!("Non compile solver used in compilation"),
        };
        ac.evaluate();
        if let Some(f) = args.dotfile() {
            let out = ac.as_graphviz();
            let mut outfile = File::create(f).unwrap();
            match outfile.write_all(out.as_bytes()) {
                Ok(_) => (),
                Err(e) => println!("Could not write the circuit into the dot file: {:?}", e),
            }
        }
        ac.solution()
    };
    if !args.statistics() {
        solution.print();
    }
    solution.to_f64()
}

pub fn pysearch(args: Args, distributions: &[Vec<f64>], clauses: &[Vec<isize>]) -> (f64, f64) {
    let mut state = StateManager::default();
    let propagator = Propagator::new(&mut state);
    let distributions_rational = distributions.iter().map(|d| d.iter().map(|f| rational(*f)).collect::<Vec<Rational>>()).collect::<Vec<Vec<Rational>>>();
    let problem = create_problem(&distributions_rational, clauses, &mut state);
    let caching_scheme = CachingScheme::new(args.caching());
    let component_extractor = ComponentExtractor::new(&problem, caching_scheme, &mut state);
    let solver = generic_solver(problem, state, component_extractor, propagator, &args);
    let parameters = SolverParameters::new(&args);
    let solution = match solver {
        GenericSolver::Search(mut solver) => solver.search(&parameters),
        GenericSolver::LogSearch(mut solver) => solver.search(&parameters),
        _ => panic!("Non search solver used in search"),
    };
    if !args.statistics() {
        solution.print();
    }
    solution.bounds()
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
    Search(Solver<false, false>),
    LogSearch(Solver<true, false>),
    Compiler(Solver<false, true>),
    LogCompiler(Solver<true, true>),
}

pub fn generic_solver(problem: Problem, state: StateManager, component_extractor: ComponentExtractor, propagator: Propagator, args: &Args) -> GenericSolver {
    let branching: Box<dyn BranchingDecision> = match args.branching() {
        Branching::MinInDegree => Box::<MinInDegree>::default(),
        Branching::MinOutDegree => Box::<MinOutDegree>::default(),
        Branching::DLCS => Box::<DLCS>::default(),
        Branching::DLCSVar => Box::<DLCSVar>::default(),
    };
    if !args.compile() && !args.statistics() {
        let solver = Solver::<false, false>::new(problem, state, component_extractor, branching, propagator);
        GenericSolver::Search(solver)
    } else if !args.compile() && args.statistics() {
        let solver = Solver::<true, false>::new(problem, state, component_extractor, branching, propagator);
        GenericSolver::LogSearch(solver)
    } else if args.compile() && !args.statistics() {
        let solver = Solver::<false, true>::new(problem, state, component_extractor, branching, propagator);
        GenericSolver::Compiler(solver)
    } else {
        let solver = Solver::<true, true>::new(problem, state, component_extractor, branching, propagator);
        GenericSolver::LogCompiler(solver)
    }
}
