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
mod semiring;

use std::ffi::OsString;
use std::path::PathBuf;
use std::fs::File;
use std::io::{Write, BufRead, BufReader};
use rug::Float;
use clap::{Parser, Subcommand};

use learning::{Learning, learner::Learner, LearnParameters};
use ac::ac::Dac;
use search_trail::StateManager;

use core::components::ComponentExtractor;
use core::problem::Problem;
use parsers::*;

use propagator::Propagator;
use common::*;
use branching::*;

pub use solver::Solver;
use solver::SolverParameters;

use peak_alloc::PeakAlloc;
#[global_allocator]
pub static PEAK_ALLOC: PeakAlloc = PeakAlloc;

#[derive(Parser)]
#[clap(name="Schlandals", version, author, about)]
pub struct Args {
    /// The input file
    #[clap(short, long, value_parser)]
    input: PathBuf,
    /// Evidence file, containing the query
    #[clap(long)]
    evidence: Option<OsString>,
    #[clap(long,short, default_value_t=u64::MAX)]
    timeout: u64,
    /// How to branch
    #[clap(short, long, value_enum, default_value_t=Branching::MinInDegree)]
    branching: Branching,
    /// Collect stats during the search, default no
    #[clap(long, action)]
    statistics: bool,
    /// The memory limit, in mega-bytes
    #[clap(short, long, default_value_t=u64::MAX)]
    memory: u64,
    /// Epsilon, the quality of the approximation (must be between greater or equal to 0). If 0 or absent, performs exact search
    #[clap(short, long, default_value_t=0.0)]
    epsilon: f64,
    /// If epsilon present, use the appropriate approximate method
    #[clap(short, long, value_enum, default_value_t=ApproximateMethod::Bounds)]
    approx: ApproximateMethod,
    #[clap(subcommand)]
    pub subcommand: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    Compile {
        /// If the problem is compiled, store it in this file
        #[clap(long)]
        fdac: Option<PathBuf>,
        /// If the problem is compiled, store a DOT graphical representation in this file
        #[clap(long)]
        dotfile: Option<PathBuf>,
    },
    Learn {
        /// The csv file containing the test filenames and the associated expected output
        #[clap(long, value_parser, value_delimiter=' ')]
        testfile: Option<PathBuf>,
        /// If present, folder in which to store the output files of the learning
        #[clap(long)]
        outfolder: Option<PathBuf>,
        /// Learning rate
        #[clap(short, long, default_value_t=0.3)]
        lr: f64,
        /// Number of epochs
        #[clap(long, default_value_t=6000)]
        nepochs: usize,
        /// If present, save a detailled csv of the training and use a codified output filename
        #[clap(long, short, action)]
        do_log: bool,
        /// If present, define the learning timeout
        #[clap(long, default_value_t=u64::MAX)]
        ltimeout: u64,
        /// Loss to use for the training, default is the MAE
        /// Possible values: MAE, MSE
        #[clap(long, default_value_t=Loss::MAE, value_enum)]
        loss: Loss, 
        /// Number of threads to use for the evaluation of the DACs
        #[clap(long, default_value_t=1, short)]
        jobs: usize,
        /// The semiring on which to evaluate the circuits. If `tensor`, use torch
        /// to compute the gradients. If `probability`, use custom efficient backpropagations
        #[clap(long, short, default_value_t=Semiring::Probability, value_enum)]
        semiring: Semiring,
        /// The optimizer to use if `tensor` is selected as semiring
        #[clap(long, short, default_value_t=Optimizer::Adam, value_enum)]
        optimizer: Optimizer,
        /// The drop in the learning rate to apply at each step
        #[clap(long, default_value_t=0.75)]
        lr_drop: f64,
        /// The number of epochs after which to drop the learning rate
        /// (i.e. the learning rate is multiplied by `lr_drop`)
        #[clap(long, default_value_t=100)]
        epoch_drop: usize,
        /// The stopping criterion for the training
        /// (i.e. if the loss is below this value, stop the training)
        #[clap(long, default_value_t=0.0001)]
        early_stop_threshold: f64,
        /// The minimum of improvement in the loss to consider that the training is still improving
        /// (i.e. if the loss is below this value for a number of epochs, stop the training)
        #[clap(long, default_value_t=0.00001)]
        early_stop_delta: f64,
        /// The number of epochs to wait before stopping the training if the loss is not improving
        /// (i.e. if the loss is below this value for a number of epochs, stop the training)
        #[clap(long, default_value_t=5)]
        patience: usize,
    },
}

impl Args {

    fn solver_param(&self) -> SolverParameters {
        SolverParameters::new(self.memory, self.epsilon, self.timeout)
    }
}

pub fn search(args: Args) -> f64 {
    let parameters = args.solver_param();
    let mut state = StateManager::default();
    let propagator = Propagator::new(&mut state);
    let problem = problem_from_input(&args.input, &args.evidence, &mut state);
    let component_extractor = ComponentExtractor::new(&problem, &mut state);
    let solver = generic_solver(problem, state, component_extractor, args.branching, propagator, parameters, args.statistics);

    let solution = match args.approx {
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

pub fn compile(args: Args) -> f64 {
    let parameters = args.solver_param();
    let mut state = StateManager::default();
    let propagator = Propagator::new(&mut state);
    let problem = problem_from_input(&args.input, &args.evidence, &mut state);
    let component_extractor = ComponentExtractor::new(&problem, &mut state);
    let solver = generic_solver(problem, state, component_extractor, args.branching, propagator, parameters, args.statistics);

    let ac: Dac<Float> = match args.approx {
        ApproximateMethod::Bounds => {
            match solver {
                GenericSolver::SMinInDegree(mut solver) => solver.compile(false),
                GenericSolver::QMinInDegree(mut solver) => solver.compile(false),
            }
        },
        ApproximateMethod::LDS => {
            match solver {
                GenericSolver::SMinInDegree(mut solver) => solver.compile(true),
                GenericSolver::QMinInDegree(mut solver) => solver.compile(true),
            }
        },
    };
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

pub type CsvInput = (PathBuf, Option<OsString>);

pub fn make_learner(inputs: Vec<CsvInput>, expected: Vec<f64>, epsilon: f64, approx:ApproximateMethod, branching: Branching, outfolder: Option<PathBuf>, jobs: usize, log: bool, semiring: Semiring, params: &LearnParameters, test_inputs:Vec<CsvInput>, test_expected:Vec<f64>) -> Box<dyn Learning> {
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

fn parse_csv(filename: PathBuf) -> (Vec<CsvInput>, Vec<f64>) {
    let mut inputs: Vec<CsvInput> = vec![];
    let mut expected: Vec<f64> = vec![];
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    for line in reader.lines().skip(1) {
        let l = line.unwrap();
        let split = l.split(',').collect::<Vec<&str>>();
        if split.len() == 3 {
            inputs.push((split[0].parse::<PathBuf>().unwrap(), Some(split[1].parse::<OsString>().unwrap())));
            expected.push(split[2].parse::<f64>().unwrap());
        } else {
            inputs.push((split[0].parse::<PathBuf>().unwrap(), None));
            expected.push(split[1].parse::<f64>().unwrap());
        }
    }
    (inputs, expected)
}

pub fn learn(args: Args) {
    if let Some(Command::Learn { testfile,
                    outfolder,
                    lr,
                    nepochs,
                    do_log,
                    ltimeout,
                    loss,
                    jobs,
                    semiring,
                    optimizer,
                    lr_drop,
                    epoch_drop,
                    early_stop_threshold,
                    early_stop_delta,
                    patience, }) = args.subcommand {
        let (inputs, expected) = parse_csv(args.input);
        let (test_inputs, test_expected) = if let Some(testfile) = testfile { parse_csv(testfile) } else { (vec![], vec![]) };
        let params = LearnParameters::new(
            lr,
            nepochs,
            args.timeout,
            ltimeout,
            loss,
            optimizer,
            lr_drop,
            epoch_drop,
            early_stop_threshold,
            early_stop_delta,
            patience,
        );
        let mut learner = make_learner(inputs, expected, args.epsilon, args.approx, args.branching, outfolder, jobs, do_log, semiring, &params, test_inputs, test_expected);
        learner.train(&params);
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
