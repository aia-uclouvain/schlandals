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
mod caching;

use std::ffi::OsString;
use std::path::PathBuf;
use std::fs::File;
use std::io::{Write, BufRead, BufReader};
use malachite::rational::Rational;
use clap::{Parser, Subcommand};

use learning::{learner::Learner, LearnParameters};
use ac::ac::Dac;
use search_trail::StateManager;

use core::components::ComponentExtractor;
use core::problem::Problem;
use parsers::*;

use propagator::Propagator;
pub use common::*;
use branching::*;
use caching::*;

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
    /// Caching strategy
    #[clap(short, long, value_enum, default_value_t=Caching::Hybrid)]
    pub caching: Caching,
    /// If set, deactivate the two level caching
    #[clap(short, long, default_value_t=false)]
    pub two_level_caching_deactivate: bool,
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
            caching: Caching::Hybrid,
            two_level_caching_deactivate: false,
            statistics: false,
            memory: u64::MAX,
            epsilon: 0.0,
            approx: ApproximateMethod::Bounds,
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
    },
    Learn {
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
    },
}

impl Args {

    fn solver_param(&self) -> SolverParameters {
        SolverParameters::new(self.memory, self.epsilon, self.timeout)
    }
}

impl Command {

    pub fn default_learn_args() -> Self {
        Self::Learn {
            trainfile: PathBuf::default(),
            testfile: None,
            outfolder: None,
            lr: 0.3,
            nepochs: 6000,
            do_log: false,
            ltimeout: u64::MAX,
            loss: Loss::MAE,
            jobs: 1,
            optimizer: Optimizer::Adam,
            lr_drop: 0.75,
            epoch_drop:100,
            early_stop_threshold: 0.0001,
            early_stop_delta: 0.00001,
            patience: 5,
            equal_init: false,
            recompile: false,
            e_weighted: false,
        }
    }
}

pub fn search(args: Args) -> f64 {
    let parameters = args.solver_param();
    let mut state = StateManager::default();
    let propagator = Propagator::new(&mut state);
    let parser = parser_from_input(args.input.clone(), args.evidence.clone());
    let problem = parser.problem_from_file(&mut state);
    let caching_scheme = CachingScheme::new(!args.two_level_caching_deactivate, args.caching);
    let component_extractor = ComponentExtractor::new(&problem, caching_scheme, &mut state);
    let solver = generic_solver(problem, state, component_extractor, args.branching, propagator, parameters, args.statistics, false);

    let solution = match args.approx {
        ApproximateMethod::Bounds => {
            match solver {
                GenericSolver::Search(mut solver) => solver.search(false),
                GenericSolver::LogSearch(mut solver) => solver.search(false),
                _ => panic!("Non search solver used in search"),
            }
        },
        ApproximateMethod::LDS => {
            match solver {
                GenericSolver::Search(mut solver) => solver.search(true),
                GenericSolver::LogSearch(mut solver) => solver.search(true),
                _ => panic!("Non search solver used in search"),
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
    let distributions_rational = distributions.iter().map(|d| d.iter().map(|f| rational(*f)).collect::<Vec<Rational>>()).collect::<Vec<Vec<Rational>>>();
    let problem = create_problem(&distributions_rational, clauses, &mut state);
    let caching_scheme = CachingScheme::new(!args.two_level_caching_deactivate, args.caching);
    let component_extractor = ComponentExtractor::new(&problem, caching_scheme, &mut state);
    let solver = generic_solver(problem, state, component_extractor, args.branching, propagator, parameters, args.statistics, false);
    let solution = match solver {
        GenericSolver::Search(mut solver) => solver.search(false),
        GenericSolver::LogSearch(mut solver) => solver.search(false),
        _ => panic!("Non search solver used in search"),
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
    let caching_scheme = CachingScheme::new(!args.two_level_caching_deactivate, args.caching);
    let component_extractor = ComponentExtractor::new(&problem, caching_scheme, &mut state);
    let solver = generic_solver(problem, state, component_extractor, args.branching, propagator, parameters, args.statistics, true);

    let mut ac: Dac = match args.approx {
        ApproximateMethod::Bounds => {
            match solver {
                GenericSolver::Search(mut solver) => solver.compile(false),
                GenericSolver::LogSearch(mut solver) => solver.compile(false),
                _ => panic!("Non compile solver used in compilation"),
            }
        },
        ApproximateMethod::LDS => {
            match solver {
                GenericSolver::Search(mut solver) => solver.compile(true),
                GenericSolver::LogSearch(mut solver) => solver.compile(true),
                _ => panic!("Non compile solver used in compilation"),
            }
        },
    };
    ac.evaluate();
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
            loss,
            optimizer,
            lr_drop,
            epoch_drop,
            early_stop_threshold,
            early_stop_delta,
            patience,
            recompile,
            e_weighted,
        );
        let approx = args.approx;
        let branching = args.branching;
        let caching = args.caching;
        let two_level_caching = !args.two_level_caching_deactivate;
        if do_log {
            let mut learner = Learner::<true>::new(args.input.clone(), args);
            learner.train(&params, branching, two_level_caching, caching, approx);
        } else {
            let mut learner = Learner::<false>::new(args.input.clone(), args);
            learner.train(&params, branching, two_level_caching, caching, approx);
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
    Search(Solver<false, false>),
    LogSearch(Solver<true, false>),
    Compiler(Solver<false, true>),
    LogCompiler(Solver<true, true>),
}

pub fn generic_solver(problem: Problem, state: StateManager, component_extractor: ComponentExtractor, branching: Branching, propagator: Propagator, parameters: SolverParameters, stat: bool, compile: bool) -> GenericSolver {
    let branching: Box<dyn BranchingDecision> = match branching {
        Branching::MinInDegree => Box::<MinInDegree>::default(),
        Branching::MinOutDegree => Box::<MinOutDegree>::default(),
        Branching::DLCS => Box::<DLCS>::default(),
        Branching::DLCSVar => Box::<DLCSVar>::default(),
    };
    if !compile && !stat {
        let solver = Solver::<false, false>::new(problem, state, component_extractor, branching, propagator, parameters);
        GenericSolver::Search(solver)
    } else if !compile && stat {
        let solver = Solver::<true, false>::new(problem, state, component_extractor, branching, propagator, parameters);
        GenericSolver::LogSearch(solver)
    } else if compile && !stat {
        let solver = Solver::<false, true>::new(problem, state, component_extractor, branching, propagator, parameters);
        GenericSolver::Compiler(solver)
    } else {
        let solver = Solver::<true, true>::new(problem, state, component_extractor, branching, propagator, parameters);
        GenericSolver::LogCompiler(solver)
    }
}
