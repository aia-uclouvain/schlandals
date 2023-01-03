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

mod core;
mod parser;
mod solver;
mod common;

use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use crate::core::components::ComponentExtractor;
use crate::core::trail::StateManager;
use parser::ppidimacs::graph_from_ppidimacs;
use solver::branching::*;
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
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Branching {
    // Averages the fiedler value of each node in a distribution
    ChildrenFiedlerAvg,
    // Takes the minimum of the fiedler value of each node in a distribution
    ChildrenFiedlerMin,
    // ChildrenFiedlerAvg with cache score
    CSChildrenFiedlerAvg,
    // ChildrenFiedlerMin with cache score
    CSChildrenFiedlerMin,
}

fn main() {
    let args = Args::parse();
    let mut state = StateManager::default();
    let (graph, prob) = graph_from_ppidimacs(&args.input, &mut state);
    match prob {
        Err(_) => println!("Initial model Unsat"),
        Ok(v) => {
            let component_extractor = ComponentExtractor::new(&graph, &mut state);
            let mut branching_heuristic: Box<dyn BranchingDecision> = match args.branching {
                Branching::ChildrenFiedlerAvg => Box::new(ChildrenFiedlerAvg::default()),
                Branching::ChildrenFiedlerMin => Box::new(ChildrenFiedlerMin::default()),
                Branching::CSChildrenFiedlerAvg => Box::new(CSChildrenFiedlerAvg::default()),
                Branching::CSChildrenFiedlerMin => Box::new(CSChildrenFiedlerMin::default()),
            };
            if args.statistics {
                let mut solver = DefaultSolver::new(
                    graph,
                    state,
                    component_extractor,
                    branching_heuristic.as_mut(),
                );
                let mut solution = solver.solve();
                solution.probability *= v;
                println!("{}", solution);
            } else {
                let mut solver = QuietSolver::new(
                    graph,
                    state,
                    component_extractor,
                    branching_heuristic.as_mut(),
                );
                let mut solution = solver.solve();
                solution.probability *= v;
                println!("{}", solution);
            }
        }
    };
}
