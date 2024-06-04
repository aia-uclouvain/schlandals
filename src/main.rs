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

use clap::{Parser, Subcommand};
use schlandals::ApproximateMethod;
use std::path::PathBuf;
use std::process;
use schlandals::learning::LearnParameters;
use schlandals::solvers::Error;

#[derive(Debug, Parser)]
#[clap(name="Schlandals", version, author, about)]
pub struct App {
    #[clap(subcommand)]
    command: Command,
    #[clap(long,short)]
    timeout: Option<u64>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// DPLL-style search based solver.
    Search {
        /// The input file
        #[clap(short, long, value_parser)]
        input: PathBuf,
        /// How to branch
        #[clap(short, long, value_enum, default_value_t=schlandals::Branching::MinInDegree)]
        branching: schlandals::Branching,
        /// Collect stats during the search, default yes
        #[clap(short, long, action)]
        statistics: bool,
        /// The memory limit, in mega-bytes
        #[clap(short, long)]
        memory: Option<u64>,
        /// Epsilon, the quality of the approximation (must be between greater or equal to 0). If 0 or absent, performs exact search
        #[clap(short, long, default_value_t=0.0)]
        epsilon: f64,
        /// If epsilon present, use the appropriate approximate method
        #[clap(short, long, value_enum, default_value_t=ApproximateMethod::Bounds)]
        approx: ApproximateMethod,
    },
    /// Use the DPLL-search structure to produce an arithmetic circuit for the problem
    Compile {
        /// The input file
        #[clap(short, long, value_parser)]
        input: PathBuf,
        /// How to branch
        #[clap(short, long, value_enum, default_value_t=schlandals::Branching::MinInDegree)]
        branching: schlandals::Branching,
        /// If present, store a textual representation of the compiled circuit
        #[clap(long)]
        fdac: Option<PathBuf>,
        /// If present, store a DOT representation of the compiled circuit
        #[clap(long)]
        dotfile: Option<PathBuf>,
        /// Epsilon, the quality of the approximation (must be between greater or equal to 0). If 0 or absent, performs exact search
        #[clap(short, long)]
        epsilon: Option<f64>,
    },
    /// Learn distribution parameters from a set of queries
    Learn {
        /// The csv file containing the cnf filenames and the associated expected output
        #[clap(long, value_parser, value_delimiter=' ')]
        trainfile: PathBuf,
        /// The csv file containing the test cnf filenames and the associated expected output
        #[clap(long, value_parser, value_delimiter=' ')]
        testfile: Option<PathBuf>,
        /// How to branch
        #[clap(short, long, value_enum, default_value_t=schlandals::Branching::MinInDegree)]
        branching: schlandals::Branching,
        /// If present, folder in which to store the output files
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
        /// If present, the epsilon used for the approximation. Value set by default to 0, thus performing exact search
        #[clap(short, long, default_value_t=0.0)]
        epsilon: f64,
        /// Loss to use for the training, default is the MAE
        /// Possible values: MAE, MSE
        #[clap(long, default_value_t=schlandals::Loss::MAE, value_enum)]
        loss: schlandals::Loss, 
        /// Number of threads to use for the evaluation of the DACs
        #[clap(long, default_value_t=1, short)]
        jobs: usize,
        /// The semiring on which to evaluate the circuits. If `tensor`, use torch
        /// to compute the gradients. If `probability`, use custom efficient backpropagations
        #[clap(long, short, default_value_t=schlandals::Semiring::Probability, value_enum)]
        semiring: schlandals::Semiring,
        /// The optimizer to use if `tensor` is selected as semiring
        #[clap(long, short, default_value_t=schlandals::Optimizer::Adam, value_enum)]
        optimizer: schlandals::Optimizer,
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
        /// If present, where to save the compiled circuits as fdac files
        #[clap(long, short)]
        save_fdac: Option<PathBuf>,
    },
    /// Partial approx
    Partial{
        /// The csv file containing the cnf filenames and the associated expected output
        #[clap(long, value_parser, value_delimiter=' ')]
        trainfile: PathBuf,
        /// The csv file containing the test cnf filenames and the associated expected output
        #[clap(long, value_parser, value_delimiter=' ')]
        testfile: Option<PathBuf>,
        /// How to branch
        #[clap(short, long, value_enum, default_value_t=schlandals::Branching::MinInDegree)]
        branching: schlandals::Branching,
        /// If present, folder in which to store the output files
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
        /// If present, the epsilon used for the approximation. Value set by default to 0, thus performing exact search
        #[clap(short, long, default_value_t=0.0)]
        epsilon: f64,
        /// Loss to use for the training, default is the MAE
        /// Possible values: MAE, MSE
        #[clap(long, default_value_t=schlandals::Loss::MAE, value_enum)]
        loss: schlandals::Loss, 
        /// Number of threads to use for the evaluation of the DACs
        #[clap(long, default_value_t=1, short)]
        jobs: usize,
        /// The semiring on which to evaluate the circuits. If `tensor`, use torch
        /// to compute the gradients. If `probability`, use custom efficient backpropagations
        #[clap(long, short, default_value_t=schlandals::Semiring::Probability, value_enum)]
        semiring: schlandals::Semiring,
        /// The optimizer to use if `tensor` is selected as semiring
        #[clap(long, short, default_value_t=schlandals::Optimizer::Adam, value_enum)]
        optimizer: schlandals::Optimizer,
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
        /// If present, where to save the compiled circuits as fdac files
        #[clap(long)]
        save_fdac: Option<PathBuf>,
        /// If not zero, the delta value to perform epsilon-delta approximation
        #[clap(long, default_value_t=0.0)]
        delta: f64,
        /// If true use sampling approximation
        #[clap(long, action)]
        sampling: bool,
    }
}

fn main() {
    let app = App::parse();
    let timeout = match app.timeout {
        Some(t) => t,
        None => u64::MAX,
    };
    match app.command {
        Command::Search { input, branching, statistics, memory , epsilon, approx} => {
            match schlandals::search(input, branching, statistics, memory, epsilon, approx, timeout) {
                Err(e) => {
                    match e {
                        Error::Unsat => println!("Model UNSAT"),
                        Error::Timeout => println!("Timeout"),
                    };
                },
                Ok(_p) => (), //println!("{}", p),
            };
        },
        Command::Compile { input, branching, fdac, dotfile, epsilon} => {
            let e = match epsilon {
                Some(v) => v,
                None => 0.0,
            };
            match schlandals::compile(input, branching, fdac, dotfile, e, timeout) {
                Err(e) => {
                    match e {
                        Error::Unsat => println!("Model UNSAT"),
                        Error::Timeout => println!("Timeout"),
                    };
                },
                Ok(_p) => {},//println!("{}", p),
            };
        },
        Command::Learn { trainfile, testfile, branching, outfolder, lr, nepochs, 
            do_log , ltimeout, epsilon, loss, jobs, semiring, optimizer, lr_drop, 
            epoch_drop, early_stop_threshold, early_stop_delta, patience, save_fdac} => {
            let params = LearnParameters::new(
                lr,
                nepochs,
                timeout,
                ltimeout,
                loss,
                optimizer,
                lr_drop,
                epoch_drop,
                early_stop_threshold,
                early_stop_delta,
                patience,
            );
            if do_log && outfolder.is_none() {
                eprintln!("Error: if do-log is set, then outfolder should be specified");
                process::exit(1);
            }
            schlandals::learn(trainfile, testfile, branching, outfolder, do_log, epsilon, jobs, semiring, params, save_fdac);
        },
        Command::Partial { trainfile, testfile, branching, outfolder, lr, nepochs, 
            do_log, ltimeout, epsilon, loss, jobs, semiring, optimizer, lr_drop, 
            epoch_drop, early_stop_threshold, early_stop_delta, patience , save_fdac, delta, sampling} => {
            let params = LearnParameters::new(
                lr,
                nepochs,
                timeout,
                ltimeout,
                loss,
                optimizer,
                lr_drop,
                epoch_drop,
                early_stop_threshold,
                early_stop_delta,
                patience,
            );
            if do_log && outfolder.is_none() {
                eprintln!("Error: if do-log is set, then outfolder should be specified");
                process::exit(1);
            }
            schlandals::partial(trainfile, testfile, branching, outfolder, do_log, epsilon, jobs, semiring, params, save_fdac, delta, sampling);
        }
    }
}
