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
use std::path::PathBuf;


#[derive(Debug, Parser)]
#[clap(name="Schlandals", version, author, about)]
pub struct App {
    #[clap(subcommand)]
    command: Command,
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
        #[clap(short, long)]
        epsilon: Option<f64>,
    },
    /// Use the DPLL-search structure to produce an arithmetic circuit for the problem
    Compile {
        /// The input file
        #[clap(short, long, value_parser)]
        input: PathBuf,
        /// How to branch
        #[clap(short, long, value_enum, default_value_t=schlandals::Branching::MinInDegree)]
        branching: schlandals::Branching,
        /// The ratio of distribution to branch on before stopping the compilation of a branch.
        /// By default make a full compilation
        #[clap(long, default_value_t=1.0)]
        ratio: f64,
        /// If present, store a textual representation of the compiled circuit
        #[clap(long)]
        fdac: Option<PathBuf>,
        /// If present, store a DOT representation of the compiled circuit
        #[clap(long)]
        dotfile: Option<PathBuf>,
    },
    /// Learn a circuit from a set of queries
    Learn {
        /// The csv file containing the cnf filenames and the associated expected output
        #[clap(long, value_parser, num_args=1.., value_delimiter=' ')]
        trainfile: PathBuf,
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
        #[clap(long, default_value_t=2000)]
        nepochs: usize,
        /// If present, save a detailled csv of the training and use a codified output filename
        #[clap(long, short, action)]
        do_log: bool,
        /// If present, define the learning timeout
        #[clap(long, default_value_t=i64::MAX)]
        timeout: i64,
        /// If present, do an approximate learner on the given ratio of distributions.
        #[clap(long, default_value_t=1.0)]
        rlearned: f64,
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
    }
}

fn main() {
    let app = App::parse();
    match app.command {
        Command::Search { input, branching, statistics, memory , epsilon} => {
            let e = match epsilon {
                Some(v) => v,
                None => 0.0,
            };
            match schlandals::search(input, branching, statistics, memory, e) {
                Err(_) => println!("Model UNSAT"),
                Ok(p) => println!("{}", p),
            };
        },
        Command::Compile { input, branching, ratio, fdac, dotfile} => {
            schlandals::compile(input, branching, ratio, fdac, dotfile);
        },
        Command::Learn { trainfile, branching, outfolder, lr, nepochs, do_log , timeout, rlearned, epsilon, loss, jobs} => {
            schlandals::learn(trainfile, branching, outfolder, lr, nepochs, do_log, timeout, rlearned, epsilon, loss, jobs);
        }
    }
}
