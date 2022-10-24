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

use crate::core::components::{ComponentExtractor, DFSComponentExtractor, NoComponentExtractor};
use crate::core::trail::StateManager;
use parser::ppidimacs::graph_from_ppidimacs;
use solver::branching::*;
use solver::sequential::Solver;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, value_parser)]
    input: PathBuf,
    /// How to detect components
    #[clap(short, long, value_enum)]
    cextractor: CExtractor,
    // How to branch
    #[clap(short, long, value_enum)]
    branching: Branching,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CExtractor {
    /// Extract component using a DFS
    Dfs,
    /// Do not detected components
    NoExtractor,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Branching {
    // Linear branching on the distributions
    First,
    // Active degree of the disributions
    ActiveDegree,
}

fn main() {
    let args = Args::parse();
    let mut state = StateManager::default();
    match graph_from_ppidimacs(&args.input, &mut state) {
        Err(_) => println!("Initial model Unsat"),
        Ok((graph, v)) => {
            let mut component_extractor: Box<dyn ComponentExtractor> = match args.cextractor {
                CExtractor::Dfs => Box::new(DFSComponentExtractor::new(&graph, &mut state)),
                CExtractor::NoExtractor => Box::new(NoComponentExtractor::new(&graph)),
            };
            let mut branching_heuristic: Box<dyn BranchingDecision> = match args.branching {
                Branching::First => Box::new(FirstBranching::default()),
                Branching::ActiveDegree => Box::new(ActiveDegreeBranching::default()),
            };
            let mut solver = Solver::new(
                graph,
                state,
                component_extractor.as_mut(),
                branching_heuristic.as_mut(),
            );
            println!("Input file {:?}", args.input);
            let value = solver.solve(v);
            println!("Solution is {} (prob {})", value, 2_f64.powf(value));
        }
    };
}
