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

use learning::{learner::Learner, LearnParameters};
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
    pub input: PathBuf,
    /// Evidence file, containing the query
    #[clap(long, required=false)]
    pub evidence: Option<OsString>,
    #[clap(long,short, default_value_t=u64::MAX)]
    pub timeout: u64,
    /// How to branch
    #[clap(short, long, value_enum, default_value_t=Branching::MinInDegree)]
    pub branching: Branching,
    /// Collect stats during the search, default no
    #[clap(long, action)]
    pub statistics: bool,
    /// The memory limit, in mega-bytes
    #[clap(short, long, default_value_t=u64::MAX)]
    pub memory: u64,
    /// Epsilon, the quality of the approximation (must be between greater or equal to 0). If 0 or absent, performs exact search
    #[clap(short, long, default_value_t=0.0)]
    pub epsilon: f64,
    /// If epsilon present, use the appropriate approximate method
    #[clap(short, long, value_enum, default_value_t=ApproximateMethod::Bounds)]
    pub approx: ApproximateMethod,
    #[clap(subcommand)]
    pub subcommand: Option<Command>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            input: PathBuf::default(),
            evidence: None,
            timeout: u64::MAX,
            branching: Branching::MinInDegree,
            statistics: false,
            memory: u64::MAX,
            epsilon: 0.0,
            approx: ApproximateMethod::LDS,
            subcommand: None,
        }
    }
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
        /// Compilation method between models (default) and non-models
        #[clap(long, default_value_t=CompilationMethod::Models, value_enum)]
        compilation_m: CompilationMethod,
        /// If lds is selected, whether to try the reduce the circuits
        #[clap(long, default_value_t=false)]
        lds_opti: bool,
    },
    Learn {
        /// The csv file containing the train filenames and the associated expected output
        #[clap(long, value_parser, value_delimiter=' ')]
        trainfile: PathBuf,
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
        /// to compute the gradients. If `probability`, use custom efficient backpropagations which is the default option
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
        /// If present, initialize the distribution weights as 1/|d|, |d| being the number of values for the distribution
        #[clap(long, action)]
        equal_init: bool,
        /// If present, recompile the circuits at each epoch
        #[clap(long, action)]
        recompile: bool,
        /// If present, weights the learning in function of the epsilon of each query
        #[clap(long, action)]
        e_weighted: bool,
        /// Indicate whether to use only the sat compilation ("models", default), or only the non-models compilation ("nonmodels") or both ("both")
        #[clap(long, default_value_t=LearningMethod::Models, value_enum)]
        learning_m: LearningMethod,
        /// If lds is selected, whether to try the reduce the circuits
        #[clap(long, default_value_t=false)]
        lds_opti: bool,
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
    let parser = parser_from_input(args.input.clone(), args.evidence.clone());
    let problem = parser.problem_from_file(&mut state);
    let component_extractor = ComponentExtractor::new(&problem, &mut state);
    let solver = generic_solver(problem, state, component_extractor, args.branching, propagator, parameters, args.statistics);

    let solution = match args.approx {
        ApproximateMethod::Bounds => {
            match solver {
                GenericSolver::SMinInDegree(mut solver) => solver.search(false,),
                GenericSolver::QMinInDegree(mut solver) => solver.search(false),
                GenericSolver::SMinConstrained(mut solver) => solver.search(false),
                GenericSolver::QMinConstrained(mut solver) => solver.search(false),
                GenericSolver::SMaxConstrained(mut solver) => solver.search(false),
                GenericSolver::QMaxConstrained(mut solver) => solver.search(false),
            }
        },
        ApproximateMethod::LDS => {
            match solver {
                GenericSolver::SMinInDegree(mut solver) => solver.search(true),
                GenericSolver::QMinInDegree(mut solver) => solver.search(true),
                GenericSolver::SMinConstrained(mut solver) => solver.search(true),
                GenericSolver::QMinConstrained(mut solver) => solver.search(true),
                GenericSolver::SMaxConstrained(mut solver) => solver.search(true),
                GenericSolver::QMaxConstrained(mut solver) => solver.search(true),
                
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
    let solver = generic_solver(problem, state, component_extractor, args.branching, propagator, parameters, args.statistics);
    let solution = match solver {
        GenericSolver::SMinInDegree(mut solver) => solver.search(false),
        GenericSolver::QMinInDegree(mut solver) => solver.search(false),
        GenericSolver::SMinConstrained(mut solver) => solver.search(false),
        GenericSolver::QMinConstrained(mut solver) => solver.search(false),
        GenericSolver::SMaxConstrained(mut solver) => solver.search(false),
        GenericSolver::QMaxConstrained(mut solver) => solver.search(false),
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
    let solver = generic_solver(problem, state, component_extractor, args.branching, propagator, parameters, args.statistics);
    let mut comp_m = CompilationMethod::Models;
    let mut lds_o = false;
    if let Command::Compile { fdac: _, dotfile: _, compilation_m , lds_opti} = args.subcommand.as_ref().unwrap() {
        comp_m = *compilation_m;
        lds_o = *lds_opti;
    }
    let (mut ac_model, mut ac_nonmodel): (Dac<Float>, Dac<Float>) = match args.approx {
        ApproximateMethod::Bounds => {
            match solver {
                GenericSolver::SMinInDegree(mut solver) => solver.compile(false, lds_o),
                GenericSolver::QMinInDegree(mut solver) => solver.compile(false, lds_o),
                GenericSolver::SMinConstrained(mut solver) => solver.compile(false, lds_o),
                GenericSolver::QMinConstrained(mut solver) => solver.compile(false, lds_o),
                GenericSolver::SMaxConstrained(mut solver) => solver.compile(false, lds_o),
                GenericSolver::QMaxConstrained(mut solver) => solver.compile(false, lds_o),
            }
        },
        ApproximateMethod::LDS => {
            match solver {
                GenericSolver::SMinInDegree(mut solver) => solver.compile(true, lds_o),
                GenericSolver::QMinInDegree(mut solver) => solver.compile(true, lds_o),
                GenericSolver::SMinConstrained(mut solver) => solver.compile(true, lds_o),
                GenericSolver::QMinConstrained(mut solver) => solver.compile(true, lds_o),
                GenericSolver::SMaxConstrained(mut solver) => solver.compile(true, lds_o),
                GenericSolver::QMaxConstrained(mut solver) => solver.compile(true, lds_o),
            }
        },
    };
    println!("lower {}",ac_model.evaluate());
    println!("upper {}",ac_nonmodel.evaluate());
    let mut ac = match comp_m {
        CompilationMethod::Models => ac_model,
        CompilationMethod::NonModels => ac_nonmodel,  
    };
    ac.evaluate();
    //println!("dac\n{}", ac.as_graphviz());
    println!("Final circuit size: {}", ac.number_nodes());
    if let Some(Command::Compile { fdac, dotfile , compilation_m: _, lds_opti: _}) = args.subcommand {
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
    println!("Estimated probability {:.8} (epsilon {}) found in {} seconds", solution.to_f64(), ac.epsilon(), ac.compile_time());
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
                    semiring: _,
                    optimizer,
                    lr_drop,
                    epoch_drop,
                    early_stop_threshold,
                    early_stop_delta,
                    patience, 
                    equal_init: _,
                    recompile,
                    e_weighted,
                    learning_m: _,
                    lds_opti,}) = args.subcommand {
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
            recompile,
            e_weighted,
            lds_opti,
        );
        let approx = args.approx;
        let branching = args.branching;
        if do_log {
            let mut learner = Learner::<true>::new(args.input.clone(), args);
            learner.train(&params, branching, approx);
        } else {
            let mut learner = Learner::<false>::new(args.input.clone(), args);
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
    SMinInDegree(Solver<MinInDegree, true>),
    QMinInDegree(Solver<MinInDegree, false>),
    SMinConstrained(Solver<MinConstrained, true>),
    QMinConstrained(Solver<MinConstrained, false>),
    SMaxConstrained(Solver<MaxConstrained, true>),
    QMaxConstrained(Solver<MaxConstrained, false>),
}

pub fn generic_solver(problem: Problem, state: StateManager, component_extractor: ComponentExtractor, branching: Branching, propagator: Propagator, parameters: SolverParameters, stat: bool) -> GenericSolver {
    if stat {
        match branching {
            Branching::MinInDegree => {
                let solver = Solver::<MinInDegree, true>::new(problem, state, component_extractor, Box::<MinInDegree>::default(), propagator, parameters);
                GenericSolver::SMinInDegree(solver)
            },
            Branching::MinConstrained => {
                let solver = Solver::<MinConstrained, true>::new(problem, state, component_extractor, Box::<MinConstrained>::default(), propagator, parameters);
                GenericSolver::SMinConstrained(solver)
            },
            Branching::MaxConstrained => {
                let solver = Solver::<MaxConstrained, true>::new(problem, state, component_extractor, Box::<MaxConstrained>::default(), propagator, parameters);
                GenericSolver::SMaxConstrained(solver)
            },
        }
    } else {
        match branching {
            Branching::MinInDegree => {
                let solver = Solver::<MinInDegree, false>::new(problem, state, component_extractor, Box::<MinInDegree>::default(), propagator, parameters);
                GenericSolver::QMinInDegree(solver)
            },
            Branching::MinConstrained => {
                let solver = Solver::<MinConstrained, false>::new(problem, state, component_extractor, Box::<MinConstrained>::default(), propagator, parameters);
                GenericSolver::QMinConstrained(solver)
            },
            Branching::MaxConstrained => {
                let solver = Solver::<MaxConstrained, false>::new(problem, state, component_extractor, Box::<MaxConstrained>::default(), propagator, parameters);
                GenericSolver::QMaxConstrained(solver)
            },
        }
    }
}
