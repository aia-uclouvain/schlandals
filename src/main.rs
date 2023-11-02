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
        #[clap(short, long, value_enum)]
        branching: schlandals::Branching,
        /// If present, store a textual representation of the compiled circuit
        #[clap(long)]
        fdac: Option<PathBuf>,
        /// If present, store a DOT representation of the compiled circuit
        #[clap(long)]
        dotfile: Option<PathBuf>,
        /// Should the DAC be read from the given input
        #[clap(short, long)]
        read: Option<bool>,
    },
    /// Learn a circuit from a set of queries
    Learn {
        /// The input files
        #[clap(short, long, value_parser)]
        inputs: PathBuf,
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
        Command::Compile { input, branching, fdac, dotfile, read} => {
            let should_read = read.is_some() && read.unwrap();
            if should_read {
                schlandals::compile(input, branching, fdac, dotfile);
            } else {
                //schlandals::read_compiled(input, dotfile);
            }
        },
        Command::Learn { inputs, branching, fout, lr, nepochs } => {
            schlandals::learn(vec![inputs], branching, fout, lr, nepochs);
        }
    }
}
