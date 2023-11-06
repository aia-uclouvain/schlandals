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

use std::path::PathBuf;
use learning::LogLearner;
use learning::QuietLearner;
use learning::exact::DACCompiler;
use sysinfo::{SystemExt, System};
use search_trail::StateManager;
use clap::ValueEnum;

use crate::core::components::ComponentExtractor;
use crate::branching::*;
use solvers::{QuietSearchSolver, StatSearchSolver};
use solvers::ProblemSolution;

use propagator::Propagator;
use crate::learning::circuit::*;

// Re-export the modules
mod common;
mod branching;
pub mod core;
mod solvers;
mod parser;
mod propagator;
mod preprocess;
mod learning;

use peak_alloc::PeakAlloc;
#[global_allocator]
pub static PEAK_ALLOC: PeakAlloc = PeakAlloc;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Branching {
    /// Minimum In-degree of a clause in the implication-graph
    MinInDegree,
    /// Minimum Out-degree of a clause in the implication-graph
    MinOutDegree,
    /// Maximum degree of a clause in the implication-graph
    MaxDegree,
    /// Variable State Independent Decaying Sum
    VSIDS,
}

pub fn compile(input: PathBuf, branching: Branching, fdac: Option<PathBuf>, dotfile: Option<PathBuf>) -> Option<Dac> {
    let mut state = StateManager::default();
    let propagator = Propagator::new(&mut state);
    let graph = parser::graph_from_ppidimacs(&input, &mut state);
    let component_extractor = ComponentExtractor::new(&graph, &mut state);
    let mut branching_heuristic: Box<dyn BranchingDecision> = match branching {
        Branching::MinInDegree => Box::<MinInDegree>::default(),
        Branching::MinOutDegree => Box::<MinOutDegree>::default(),
        Branching::MaxDegree => Box::<MaxDegree>::default(),
        Branching::VSIDS => Box::<VSIDS>::default(),
    };
    let mut compiler = DACCompiler::new(graph, state, component_extractor, branching_heuristic.as_mut(), propagator);
    let mut res = compiler.compile(u64::MAX);
    if let Some(dac) = res.as_mut() {
        dac.evaluate();
    }
    res
}

pub fn learn(inputs: Vec<PathBuf>, branching: Branching, fout: Option<PathBuf>, lr:f64, nepochs: usize, log:bool, timeout:u64) {    
    let mut state = StateManager::default();
    let graph = parser::graph_from_ppidimacs(&inputs[0], &mut state);
    let mut distributions: Vec<Vec<f64>> = vec![];
    for distribution in graph.distributions_iter() {
        let probabilities: Vec<f64>= graph[distribution].iter_variables().map(|v| graph[v].weight().unwrap()).collect();
        distributions.push(probabilities);
    }
    if log { 
        let mut learner = LogLearner::new(distributions, inputs, branching, timeout);
        learner.train(nepochs, lr, fout);
    }
    else {
        let mut learner = QuietLearner::new(distributions, inputs, branching, timeout);
        learner.train(nepochs, lr, fout);
    }
}

/* pub fn read_compiled(input: PathBuf, dotfile: Option<PathBuf>) -> Dac {
    let dac = Dac::from_file(&input);
    if let Some(f) = dotfile {
        let out = dac.as_graphviz();
        let mut outfile = File::create(f).unwrap();
        match outfile.write(out.as_bytes()) {
            Ok(_) => (),
            Err(e) => println!("Could not write the PC into the file: {:?}", e),
        }
    }
    dac
} */

pub fn search(input: PathBuf, branching: Branching, statistics: bool, memory: Option<u64>, epsilon: f64) -> ProblemSolution {
    let mut state = StateManager::default();
    let propagator = Propagator::new(&mut state);
    let graph = parser::graph_from_ppidimacs(&input, &mut state);
    let component_extractor = ComponentExtractor::new(&graph, &mut state);
    let mut branching_heuristic: Box<dyn BranchingDecision> = match branching {
        Branching::MinInDegree => Box::<MinInDegree>::default(),
        Branching::MinOutDegree => Box::<MinOutDegree>::default(),
        Branching::MaxDegree => Box::<MaxDegree>::default(),
        Branching::VSIDS => Box::<VSIDS>::default(),
    };
    let mlimit = if let Some(m) = memory {
        m
    } else {
        let sys = System::new_all();
        sys.total_memory() / 1000000
    };
    if statistics {
        let mut solver = StatSearchSolver::new(
            graph,
            state,
            component_extractor,
            branching_heuristic.as_mut(),
            propagator,
            mlimit,
            epsilon,
        );
        solver.solve()
    } else {
        let mut solver = QuietSearchSolver::new(
            graph,
            state,
            component_extractor,
            branching_heuristic.as_mut(),
            propagator,
            mlimit,
            epsilon,
        );
        solver.solve()
    }
}
