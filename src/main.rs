//Schlandals
//Copyright (C) 2022 A. Dubray
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
mod solver;

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use sysinfo::{SystemExt, System};
use search_trail::StateManager;

use crate::core::components::ComponentExtractor;
use parser::ppidimacs::graph_from_ppidimacs;
use solver::branching::*;
use solver::propagator::FTReachablePropagator;
use solver::{DefaultSolver, QuietSolver};

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
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Branching {
    /// Heuristic based on the fiedler value of the clause graph
    Fiedler,
    /// VSIDS
    Vsids,
}

fn main() {
    let args = Args::parse();
    let mut state = StateManager::default();
    let mut propagator = FTReachablePropagator::default();
    let graph = graph_from_ppidimacs(&args.input, &mut state, &mut propagator);
    let component_extractor = ComponentExtractor::new(&graph, &mut state);
    let mut branching_heuristic: Box<dyn BranchingDecision> = match args.branching {
        Branching::Fiedler => Box::<Fiedler>::default(),
        Branching::Vsids => Box::<Vsids>::default(),
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
        match solver.solve() {
            Ok(s) => println!("{}", s),
            Err(_) => println!("Model UNSAT"),
        }
    } else {
        let mut solver = QuietSolver::new(
            graph,
            state,
            component_extractor,
            branching_heuristic.as_mut(),
            propagator,
            mlimit,
        );
        match solver.solve() {
            Ok(s) => println!("{}", s),
            Err(_) => println!("Model UNSAT"),
        }
    }
}
