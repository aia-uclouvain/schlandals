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

mod common;
mod core;
mod parser;
mod search;
mod compiler;
mod heuristics;
mod propagator;

use clap::{Parser, ValueEnum, Subcommand};
use std::path::PathBuf;
use std::fs::File;
use std::io::Write;
use sysinfo::{SystemExt, System};
use search_trail::StateManager;

use crate::core::components::ComponentExtractor;
use parser::*;
use heuristics::branching::*;
use search::{DefaultSolver, QuietSolver};
use propagator::FTReachablePropagator;
use compiler::exact::ExactDACCompiler;
use compiler::circuit::DAC;

#[derive(Debug, Parser)]
#[clap(name="Schlandals", version, author, about)]
pub struct App {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Search based solver
    Search {
        /// The input file
        #[clap(short, long, value_parser)]
        input: PathBuf,
        // How to branch
        #[clap(short, long, value_enum)]
        branching: Branching,
        // Collect stats during the search, default yes
        #[clap(short, long, action)]
        statistics: bool,
        // The memory limit, in mega-bytes
        #[clap(short, long)]
        memory: Option<u64>,
    },
    Compile {
        /// The input file
        #[clap(short, long, value_parser)]
        input: PathBuf,
        // How to branch
        #[clap(short, long, value_enum)]
        branching: Branching,
        /// If present, store a textual representation of the compiled circuit
        #[clap(long)]
        fdac: Option<PathBuf>,
        // If present, store a DOT representation of the compiled circuit
        #[clap(long)]
        dotfile: Option<PathBuf>,
    },
    ReadCompiled {
        /// Reads a circuit from an input file
        #[clap(short, long, value_parser)]
        input: PathBuf,
        // If present, store a DOT representation of the compiled circuit
        #[clap(long)]
        dotfile: Option<PathBuf>,
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Branching {
    /// Heuristic based on the fiedler value of the clause graph
    Fiedler,
    /// Minimum In-degree of a clause in the implication-graph
    MinInDegree,
    /// Minimum Out-degree of a clause in the implication-graph
    MinOutDegree,
    /// Maximum degree of a clause in the implication-graph
    MaxDegree
}

fn read_compiled(input: PathBuf, dotfile: Option<PathBuf>) {
    let mut dac = DAC::from_file(&input);
    if let Some(f) = dotfile {
        let out = dac.as_graphviz();
        let mut outfile = File::create(&f).unwrap();
        match outfile.write(out.as_bytes()) {
            Ok(_) => (),
            Err(e) => println!("Culd not write the PC into the file: {:?}", e),
        }
    }
    println!("{}", dac.evaluate());
}

fn run_compilation(input: PathBuf, branching: Branching, fdac: Option<PathBuf>, dotfile: Option<PathBuf>) {
    let mut state = StateManager::default();
    let mut propagator = FTReachablePropagator::<true>::new();
    let graph = graph_from_ppidimacs(&input, &mut state, &mut propagator);
    let component_extractor = ComponentExtractor::new(&graph, &mut state);
    let mut branching_heuristic: Box<dyn BranchingDecision> = match branching {
        Branching::Fiedler => Box::<Fiedler>::default(),
        Branching::MinInDegree => Box::<MinInDegree>::default(),
        Branching::MinOutDegree => Box::<MinOutDegree>::default(),
        Branching::MaxDegree => Box::<MaxDegree>::default(),
    };
    let mut compiler = ExactDACCompiler::new(graph, state, component_extractor, branching_heuristic.as_mut(), propagator);
    match compiler.compile() {
        None => {
            println!("Model UNSAT, can not compile");       
        },
        Some(dac) => {
            println!("Compilation successfull");
            if let Some(f) = dotfile {
                let out = dac.as_graphviz();
                let mut outfile = File::create(&f).unwrap();
                match outfile.write(out.as_bytes()) {
                    Ok(_) => (),
                    Err(e) => println!("Culd not write the circuit into the dot file: {:?}", e),
                }
            }
            if let Some(f) = fdac {
                let mut outfile = File::create(&f).unwrap();
                match outfile.write(format!("{}", dac).as_bytes()) {
                    Ok(_) => (),
                    Err(e) => println!("Culd not write the circuit into the fdac file: {:?}", e),
                }
                
            }
        }
    }
}

fn run_search(input: PathBuf, branching: Branching, statistics: bool, memory: Option<u64>) {
    let mut state = StateManager::default();
    let mut propagator = FTReachablePropagator::<false>::new();
    let graph = graph_from_ppidimacs(&input, &mut state, &mut propagator);
    let component_extractor = ComponentExtractor::new(&graph, &mut state);
    let mut branching_heuristic: Box<dyn BranchingDecision> = match branching {
        Branching::Fiedler => Box::<Fiedler>::default(),
        Branching::MinInDegree => Box::<MinInDegree>::default(),
        Branching::MinOutDegree => Box::<MinOutDegree>::default(),
        Branching::MaxDegree => Box::<MaxDegree>::default(),
    };
    let mlimit = if memory.is_some() {
        memory.unwrap()
    } else {
        let sys = System::new_all();
        sys.total_memory() / 1000000
    };
    if statistics {
        let mut solver = DefaultSolver::new(
            graph,
            state,
            component_extractor,
            branching_heuristic.as_mut(),
            propagator,
            mlimit,
        );
        solver.solve();
    } else {
        let mut solver = QuietSolver::new(
            graph,
            state,
            component_extractor,
            branching_heuristic.as_mut(),
            propagator,
            mlimit,
        );
        solver.solve();
    }
}

fn main() {
    let app = App::parse();
    match app.command {
        Command::Search { input, branching, statistics, memory } => {
            run_search(input, branching, statistics, memory);
        },
        Command::Compile { input, branching, fdac, dotfile } => {
            run_compilation(input, branching, fdac, dotfile);
        }
        Command::ReadCompiled { input, dotfile } => {
            read_compiled(input, dotfile);
        }
    }
}
