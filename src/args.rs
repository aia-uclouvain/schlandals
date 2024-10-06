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
use clap::{Parser, Subcommand, ValueEnum};
use std::ffi::OsString;
use std::path::PathBuf;

use crate::branching::*;
use crate::learning::*;
use crate::ring;
use crate::solver::SolverParameters;

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
    /// If present, transform the probabilities in log space
    #[clap(long, action)]
    pub log: bool,
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
    #[clap(short, long, default_value_t=Ring::AddMul)]
    pub ring: Ring,
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
            log: false,
            branching: Branching::MinInDegree,
            statistics: false,
            memory: u64::MAX,
            epsilon: 0.0,
            ring: Ring::AddMul,
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
        /// The optimizer to use if `tensor` is selected as ring
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

    pub fn solver_param(&self) -> SolverParameters {
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Branching {
    /// Minimum In-degree of a clause in the implication-graph
    MinInDegree,
}

impl Branching {

    pub fn to_type(&self) -> impl BranchingDecision {
        match self {
            Branching::MinInDegree => MinInDegree::default(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Loss {
    MAE,
    MSE,
}

impl Loss {

    pub fn to_type(&self) -> Box<dyn LossFunctions> {
        match self {
            Loss::MAE => Box::<MAE>::default(),
            Loss::MSE => Box::<MSE>::default(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Ring {
    AddMul,
    MaxMul,
}

impl Ring {
    pub fn to_type(&self) -> Box<dyn ring::Ring> {
        match self {
            Ring::AddMul => Box::<ring::AddMulRing>::default(),
            Ring::MaxMul => Box::<ring::MaxMulRing>::default(),
        }
    }
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

impl std::fmt::Display for Ring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ring::AddMul => write!(f, "add-mul"),
            Ring::MaxMul => write!(f, "max-mul"),
        }
    }
}
