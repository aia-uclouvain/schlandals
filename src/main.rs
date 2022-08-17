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

use clap::Parser;
use std::path::PathBuf;

use crate::core::components::DFSComponentExtractor;
use crate::core::trail::TrailedStateManager;
use parser::ppidimacs::graph_from_ppidimacs;
use solver::branching::FirstBranching;
use solver::solver::Solver;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, value_parser)]
    input: PathBuf,
}

fn main() {
    let args = Args::parse();
    let mut state = TrailedStateManager::new();
    let graph = graph_from_ppidimacs(&args.input, &mut state);
    let component_extractor = DFSComponentExtractor::new(&graph, &mut state);
    let branching_heuristic = FirstBranching::default();
    let mut solver: Solver<TrailedStateManager, DFSComponentExtractor, FirstBranching> =
        Solver::new(graph, state, component_extractor, branching_heuristic);
    println!("Input file {:?}", args.input);
    let value = solver.solve();
    println!("Solution is {} (prob {})", value, 2_f64.powf(value));
}
