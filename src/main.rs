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
use propagator::FTReachablePropagator;
use search::{DefaultSolver, QuietSolver};
use compiler::exact:: ExactAOMDDCompiler;
use compiler::aomdd::AOMDD;

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
    // Should the problem be compiled into an AND/OR MDD? If true, the memory argument is not taken into account (default is ´false´)
    #[clap(short, long, default_value_t=false)]
    compiled: bool,
    // In which file to save the graphviz file of the AOMDD
    #[clap(long)]
    fgraphviz: Option<PathBuf>,
    // Read an AOMDD in the given file and evaluates it
    #[clap(long)]
    faomdd: Option<PathBuf>,
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

fn main() {
    let args = Args::parse();
    let mut state = StateManager::default();
    let mut propagator = FTReachablePropagator::default();
    let graph = graph_from_ppidimacs(&args.input, &mut state, &mut propagator);
    let component_extractor = ComponentExtractor::new(&graph, &mut state);
    let mut branching_heuristic: Box<dyn BranchingDecision> = match args.branching {
        Branching::Fiedler => Box::<Fiedler>::default(),
        Branching::MinInDegree => Box::<MinInDegree>::default(),
        Branching::MinOutDegree => Box::<MinOutDegree>::default(),
        Branching::MaxDegree => Box::<MaxDegree>::default(),
    };
    if !args.compiled {
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
    } else {
        if let Some(filepath) = args.faomdd {
            let aomdd = AOMDD::from_file(&filepath);
            println!("{}", aomdd.evaluate());
        } else {
            let mut compiler = ExactAOMDDCompiler::new(
                graph,
                state,
                component_extractor,
                branching_heuristic.as_mut(),
                propagator
            );
            let aomdd = compiler.compile();
            if let Some(path) = args.fgraphviz {
                let mut outfile = File::create(path).unwrap();
                match outfile.write(aomdd.as_graphviz().as_bytes()) {
                    Ok(_) => (),
                    Err(e) => println!("Could not write the AOMDD into the file: {:?}", e),
                };
            }
        }
    }
}
