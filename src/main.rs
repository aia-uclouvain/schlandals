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

use clap::{Parser, ValueEnum};
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

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
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
    // If given, the problem is compiled into a probabilistic circuit and the memory limit is not taken into account
    #[clap(short, long, action)]
    compiled: bool,
    // In which file to store/read the compiled probabilistic circuit
    #[clap(long)]
    fpc: Option<PathBuf>,
    #[clap(short, long, action)]
    // If present, the compiler reads from the given file and evaluate the circuits
    read_compiled: bool,
    // Store a DOT representation of the compiled circuit
    #[clap(long)]
    dotfile: Option<PathBuf>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
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

fn run_compilation(args: &Args) {
    if args.read_compiled {
        let mut spn = DAC::from_file(args.fpc.as_ref().unwrap());
        println!("{:?}", spn.evaluate());
    } else {
        let mut state = StateManager::default();
        let mut propagator = FTReachablePropagator::<true>::new();
        let graph = graph_from_ppidimacs(&args.input, &mut state, &mut propagator);
        let component_extractor = ComponentExtractor::new(&graph, &mut state);
        let mut branching_heuristic: Box<dyn BranchingDecision> = match args.branching {
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
            Some(pc) => {
                println!("Compilation successfull");
                if args.dotfile.is_some() {
                    let out = pc.as_graphviz();
                    let mut outfile = File::create(args.dotfile.as_ref().unwrap()).unwrap();
                    match outfile.write(out.as_bytes()) {
                        Ok(_) => (),
                        Err(e) => println!("Culd not write the PC into the file: {:?}", e),
                    }
                }
                if args.fpc.is_some() {
                    let mut outfile = File::create(args.fpc.as_ref().unwrap()).unwrap();
                    match outfile.write(format!("{}", pc).as_bytes()) {
                        Ok(_) => (),
                        Err(e) => println!("Culd not write the PC into the file: {:?}", e),
                    }
                    
                }
            }
        }

    }
}

fn run_search(args: &Args) {
    let mut state = StateManager::default();
    let mut propagator = FTReachablePropagator::<false>::new();
    let graph = graph_from_ppidimacs(&args.input, &mut state, &mut propagator);
    let component_extractor = ComponentExtractor::new(&graph, &mut state);
    let mut branching_heuristic: Box<dyn BranchingDecision> = match args.branching {
        Branching::Fiedler => Box::<Fiedler>::default(),
        Branching::MinInDegree => Box::<MinInDegree>::default(),
        Branching::MinOutDegree => Box::<MinOutDegree>::default(),
        Branching::MaxDegree => Box::<MaxDegree>::default(),
    };
    let mlimit = if args.memory.is_some() {
        args.memory.unwrap()
    } else {
        let sys = System::new_all();
        sys.total_memory() / 1000000
    };
    if args.statistics {
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
    let args = Args::parse();
    if args.compiled {
        run_compilation(&args);
    } else {
        run_search(&args);
    }
}
