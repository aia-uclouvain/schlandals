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
        #[clap(short, long, value_enum)]
        branching: schlandals::Branching,
        /// Collect stats during the search, default yes
        #[clap(short, long, action)]
        statistics: bool,
        /// The memory limit, in mega-bytes
        #[clap(short, long)]
        memory: Option<u64>,
    },
    /// Approximate DPLL-style solver providing epsilon guarantees on the approximation
    ApproximateSearch {
        /// The input file
        #[clap(short, long, value_parser)]
        input: PathBuf,
        /// How to branch
        #[clap(short, long, value_enum)]
        branching: schlandals::Branching,
        /// Collect stats during the search, default yes
        #[clap(short, long, action)]
        statistics: bool,
        /// The memory limit, in mega-bytes
        #[clap(short, long)]
        memory: Option<u64>,
        /// Epsilon, the quality of the approximation (must be between 0 and 1, inclusive)
        #[clap(short, long)]
        epsilon: f64,
    },
    /// Use the DPLL-search structure to produce an arithmetic circuit for the problem
    Compile {
        /// The input file
        #[clap(short, long, value_parser)]
        input: PathBuf,
        /// How to branch
        #[clap(short, long, value_enum)]
        branching: schlandals::Branching,
        /// If present, store a textual representation of the compiled circuit
        #[clap(long)]
        fdac: Option<PathBuf>,
        /// If present, store a DOT representation of the compiled circuit
        #[clap(long)]
        dotfile: Option<PathBuf>,
    },
    /// Read and evaluate an arithmetic circuits that was previously created with the compile sub-command
    ReadCompiled {
        /// Reads a circuit from an input file
        #[clap(short, long, value_parser)]
        input: PathBuf,
        /// If present, store a DOT representation of the compiled circuit
        #[clap(long)]
        dotfile: Option<PathBuf>,
    },
    /// Learn a circuit from a set of queries
    Learn {
        /// The input files
        #[clap(short, long, value_parser, num_args=1.., value_delimiter=' ')]
        inputs: Vec<PathBuf>,
        /// How to branch
        #[clap(short, long, value_enum)]
        branching: schlandals::Branching,
        /// If present, file to store the learned distributions
        #[clap(long)]
        fout: Option<PathBuf>,
        /// Learning rate
        #[clap(short, long)]
        lr: f64,
        /// Number of epochs
        #[clap(short, long)]
        nepochs: usize,
        /// If present, save a detailled csv of the training and use a codified output filename
        #[clap(long, short, action)]
        do_log: bool,
        /// If present, define the compilation timeout
        #[clap(long, short, default_value_t=u64::MAX)]
        timeout: u64,
    }
}

fn main() {
    let app = App::parse();
    match app.command {
        Command::Search { input, branching, statistics, memory } => {
            match schlandals::search(input, branching, statistics, memory, 0.0) {
                Err(_) => println!("Model UNSAT"),
                Ok(p) => println!("{}", p),
            };
        },
        Command::Compile { input, branching, fdac, dotfile } => {
            schlandals::compile(input, branching, fdac, dotfile);
        },
        Command::ReadCompiled { input, dotfile } => {
            //schlandals::read_compiled(input, dotfile);
        },
        Command::ApproximateSearch { input, branching, statistics, memory, epsilon }  => {
            match schlandals::search(input, branching, statistics, memory, epsilon) {
                Err(_) => println!("Model UNSAT"),
                Ok(p) => println!("{}", p),
            };
        }
        Command::Learn { inputs, branching, fout, lr, nepochs, do_log , timeout} => {
            schlandals::learn(inputs, branching, fout, lr, nepochs, do_log, timeout);
        }
    }
}
