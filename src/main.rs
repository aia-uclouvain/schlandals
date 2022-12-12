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
    // Number of articulation point in the distribution
    Articulation,
    // Size of the distribution
    Size,
    // Neighbor fiedler
    NeighborFiedler,
}

fn main() {
    let args = Args::parse();
    let mut state = StateManager::default();
    match graph_from_ppidimacs(&args.input, &mut state) {
        Err(_) => println!("Initial model Unsat"),
        Ok((graph, v)) => {
            let component_extractor = ComponentExtractor::new(&graph, &mut state);
            let mut branching_heuristic: Box<dyn BranchingDecision> = match args.branching {
                Branching::Articulation => Box::new(Articulation::default()),
                Branching::Size => Box::new(Size::default()),
                Branching::NeighborFiedler => Box::new(NeighborDiffFiedler::default()),
            };
            if args.statistics {
                let mut solver = DefaultSolver::new(
                    graph,
                    state,
                    component_extractor,
                    branching_heuristic.as_mut(),
                );
                let mut solution = solver.solve();
                solution.probability += v;
                println!("{}", solution);
            } else {
                let mut solver = QuietSolver::new(
                    graph,
                    state,
                    component_extractor,
                    branching_heuristic.as_mut(),
                );
                let mut solution = solver.solve();
                solution.probability += v;
                println!("{}", solution);
            }
        }
    };
}
