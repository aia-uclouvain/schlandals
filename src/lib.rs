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

// Re-export the modules
mod solver;
mod statistics;
pub mod common;
mod branching;
pub mod core;
mod parsers;
mod propagator;
mod preprocess;
pub mod learning;
pub mod ac;
pub mod ring;
pub mod args;

use std::ffi::OsString;
use std::path::PathBuf;
use std::fs::File;
use std::io::{Write, BufRead, BufReader};

use learning::{learner::Learner, LearnParameters};
use ac::ac::Dac;
use search_trail::StateManager;

use core::components::ComponentExtractor;
use core::problem::Problem;
use parsers::*;

use propagator::Propagator;
pub use common::*;
use branching::*;

pub use solver::Solver;
use solver::SolverParameters;

use args::*;

use peak_alloc::PeakAlloc;
#[global_allocator]
pub static PEAK_ALLOC: PeakAlloc = PeakAlloc;

pub fn search(args: Args) -> f64 {
    let parameters = args.solver_param();
    let mut state = StateManager::default();
    let propagator = Propagator::new(&mut state);
    let parser = parser_from_input(args.input.clone(), args.evidence.clone());
    let problem = parser.problem_from_file(&mut state);
    let component_extractor = ComponentExtractor::new(&problem, &mut state);
    let solver = generic_solver(problem, state, component_extractor, args.branching, propagator, parameters, args.statistics, false);
    let ring = args.ring.to_type();

    let solution = match args.approx {
        ApproximateMethod::Bounds => {
            match solver {
                GenericSolver::SMinInDegreeSearch(mut solver) => solver.search(false, &ring),
                GenericSolver::QMinInDegreeSearch(mut solver) => solver.search(false, &ring),
                GenericSolver::SMinInDegreeCompile(_) => panic!("Non search solver used in search"),
                GenericSolver::QMinInDegreeCompile(_) => panic!("Non search solver used in search"),
            }
        },
        ApproximateMethod::LDS => {
            match solver {
                GenericSolver::SMinInDegreeSearch(mut solver) => solver.search(true, &ring),
                GenericSolver::QMinInDegreeSearch(mut solver) => solver.search(true, &ring),
                GenericSolver::SMinInDegreeCompile(_) => panic!("Non search solver used in search"),
                GenericSolver::QMinInDegreeCompile(_) => panic!("Non search solver used in search"),
            }
        },
    };
    solution.print();
    solution.to_f64()
}

pub fn pysearch(args: Args, distributions: &[Vec<f64>], clauses: &[Vec<isize>]) -> (f64, f64) {
    let parameters = args.solver_param();
    let mut state = StateManager::default();
    let propagator = Propagator::new(&mut state);
    let problem = create_problem(distributions, clauses, &mut state);
    let component_extractor = ComponentExtractor::new(&problem, &mut state);
    let solver = generic_solver(problem, state, component_extractor, args.branching, propagator, parameters, args.statistics, false);
    let ring = args.ring.to_type();
    let solution = match solver {
        GenericSolver::SMinInDegreeSearch(mut solver) => solver.search(false, &ring),
        GenericSolver::QMinInDegreeSearch(mut solver) => solver.search(false, &ring),
        GenericSolver::SMinInDegreeCompile(_) => panic!("Non search solver used in search"),
        GenericSolver::QMinInDegreeCompile(_) => panic!("Non search solver used in search"),
    };
    solution.print();
    solution.bounds()
}

pub fn compile(args: Args) -> f64 {
    let parameters = args.solver_param();
    let mut state = StateManager::default();
    let propagator = Propagator::new(&mut state);
    let parser = parser_from_input(args.input.clone(), args.evidence.clone());
    let problem = parser.problem_from_file(&mut state);
    let component_extractor = ComponentExtractor::new(&problem, &mut state);
    let solver = generic_solver(problem, state, component_extractor, args.branching, propagator, parameters, args.statistics, true);
    let ring = args.ring.to_type();

    let mut ac: Dac = match args.approx {
        ApproximateMethod::Bounds => {
            match solver {
                GenericSolver::SMinInDegreeCompile(mut solver) => solver.compile(false, &ring),
                GenericSolver::QMinInDegreeCompile(mut solver) => solver.compile(false, &ring),
                GenericSolver::SMinInDegreeSearch(_) => panic!("Non compile solver used in compilation"),
                GenericSolver::QMinInDegreeSearch(_) => panic!("Non compile solver used in compilation"),
            }
        },
        ApproximateMethod::LDS => {
            match solver {
                GenericSolver::SMinInDegreeCompile(mut solver) => solver.compile(true, &ring),
                GenericSolver::QMinInDegreeCompile(mut solver) => solver.compile(true, &ring),
                GenericSolver::SMinInDegreeSearch(_) => panic!("Non compile solver used in compilation"),
                GenericSolver::QMinInDegreeSearch(_) => panic!("Non compile solver used in compilation"),
            }
        },
    };
    ac.evaluate(&ring);
    if let Some(Command::Compile { fdac, dotfile }) = args.subcommand {
        if let Some(f) = dotfile {
            let out = ac.as_graphviz();
            let mut outfile = File::create(f).unwrap();
            match outfile.write_all(out.as_bytes()) {
                Ok(_) => (),
                Err(e) => println!("Could not write the circuit into the dot file: {:?}", e),
            }
        }
        if let Some(f) = fdac {
            let mut outfile = File::create(f).unwrap();
            match outfile.write_all(format!("{}", ac).as_bytes()) {
                Ok(_) => (),
                Err(e) => println!("Could not write the circuit into the fdac file: {:?}", e),
            }
            
        }
    }
    let solution = ac.solution();
    solution.print();
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
    if let Some(Command::Learn { trainfile: _,
                    testfile: _,
                    outfolder: _,
                    lr,
                    nepochs,
                    do_log,
                    ltimeout,
                    loss,
                    jobs: _,
                    optimizer,
                    lr_drop,
                    epoch_drop,
                    early_stop_threshold,
                    early_stop_delta,
                    patience, 
                    equal_init: _,
                    recompile,
                    e_weighted}) = args.subcommand {
        let params = LearnParameters::new(
            lr,
            nepochs,
            args.timeout,
            ltimeout,
            loss.to_type(),
            optimizer,
            lr_drop,
            epoch_drop,
            early_stop_threshold,
            early_stop_delta,
            patience,
            recompile,
            e_weighted,
            args.ring.to_type(),
        );
        let approx = args.approx;
        let branching = args.branching;
        let ring = Box::new(args.ring.to_type());
        if do_log {
            let mut learner = Learner::<true>::new(args.input.clone(), &ring, args);
            learner.train(&params, branching, approx);
        } else {
            let mut learner = Learner::<false>::new(args.input.clone(), &ring, args);
            learner.train(&params, branching, approx);
        };
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

pub enum GenericSolver {
    SMinInDegreeSearch(Solver<MinInDegree, true, false>),
    QMinInDegreeSearch(Solver<MinInDegree, false, false>),
    SMinInDegreeCompile(Solver<MinInDegree, true, true>),
    QMinInDegreeCompile(Solver<MinInDegree, false, true>),
}

pub fn generic_solver(problem: Problem, state: StateManager, component_extractor: ComponentExtractor, branching: Branching, propagator: Propagator, parameters: SolverParameters, stat: bool, compile: bool) -> GenericSolver {
    if compile {
        if stat {
            match branching {
                Branching::MinInDegree => {
                    let solver = Solver::<MinInDegree, true, true>::new(problem, state, component_extractor, Box::<MinInDegree>::default(), propagator, parameters);
                    GenericSolver::SMinInDegreeCompile(solver)
                },
            }
        } else {
            match branching {
                Branching::MinInDegree => {
                    let solver = Solver::<MinInDegree, false, true>::new(problem, state, component_extractor, Box::<MinInDegree>::default(), propagator, parameters);
                    GenericSolver::QMinInDegreeCompile(solver)
                },
            }
        }
    } else {
        if stat {
            match branching {
                Branching::MinInDegree => {
                    let solver = Solver::<MinInDegree, true, false>::new(problem, state, component_extractor, Box::<MinInDegree>::default(), propagator, parameters);
                    GenericSolver::SMinInDegreeSearch(solver)
                },
            }
        } else {
            match branching {
                Branching::MinInDegree => {
                    let solver = Solver::<MinInDegree, false, false>::new(problem, state, component_extractor, Box::<MinInDegree>::default(), propagator, parameters);
                    GenericSolver::QMinInDegreeSearch(solver)
                },
            }
        }
    }
}
