//Schlandal
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


use rug::Float;
use crate::branching::*;


/// Unit structure representing the the problem is UNSAT
#[derive(Debug)]
pub struct Unsat;

/// Type alias used for the solution of the problem, which is either a Float or UNSAT
pub type ProblemSolution = Result<Float, Unsat>;

pub type Bounds = (Float, Float);

pub mod solver;
mod statistics;

pub use solver::Solver;
pub use sysinfo::{SystemExt, System};

pub enum GenericSolver {
    SMinInDegree(Solver<MinInDegree, true>),
    SMinOutDegree(Solver<MinOutDegree, true>),
    SMaxDegree(Solver<MaxDegree, true>),
    SVSIDS(Solver<VSIDS, true>),
    QMinInDegree(Solver<MinInDegree, false>),
    QMinOutDegree(Solver<MinOutDegree, false>),
    QMaxDegree(Solver<MaxDegree, false>),
    QVSIDS(Solver<VSIDS, false>),
}

macro_rules! make_solver {
    ($i:expr, $b:expr, $e:expr, $m:expr, $s:expr) => {
        {
            let mut state = StateManager::default();
            let propagator = Propagator::new(&mut state);
            let graph = graph_from_ppidimacs($i, &mut state, false);
            let component_extractor = ComponentExtractor::new(&graph, &mut state);
            let mlimit = if let Some(m) = $m {
                m
            } else {
                let sys = System::new_all();
                sys.total_memory() / 1000000
            };
            if $s {
                match $b {
                    Branching::MinInDegree => {
                        let solver = Solver::<MinInDegree, true>::new(graph, state, component_extractor, Box::<MinInDegree>::default(), propagator, mlimit, $e);
                        GenericSolver::SMinInDegree(solver)
                    },
                    Branching::MinOutDegree => {
                        let solver = Solver::<MinOutDegree, true>::new(graph, state, component_extractor, Box::<MinOutDegree>::default(), propagator, mlimit, $e);
                        GenericSolver::SMinOutDegree(solver)
                    },
                    Branching::MaxDegree => {
                        let solver = Solver::<MaxDegree, true>::new(graph, state, component_extractor, Box::<MaxDegree>::default(), propagator, mlimit, $e);
                        GenericSolver::SMaxDegree(solver)
                    },
                    Branching::VSIDS => {
                        let solver = Solver::<VSIDS, true>::new(graph, state, component_extractor, Box::<VSIDS>::default(), propagator, mlimit, $e);
                        GenericSolver::SVSIDS(solver)
                    },
                }
            } else {
                match $b {
                    Branching::MinInDegree => {
                        let solver = Solver::<MinInDegree, false>::new(graph, state, component_extractor, Box::<MinInDegree>::default(), propagator, mlimit, $e);
                        GenericSolver::QMinInDegree(solver)
                    },
                    Branching::MinOutDegree => {
                        let solver = Solver::<MinOutDegree, false>::new(graph, state, component_extractor, Box::<MinOutDegree>::default(), propagator, mlimit, $e);
                        GenericSolver::QMinOutDegree(solver)
                    },
                    Branching::MaxDegree => {
                        let solver = Solver::<MaxDegree, false>::new(graph, state, component_extractor, Box::<MaxDegree>::default(), propagator, mlimit, $e);
                        GenericSolver::QMaxDegree(solver)
                    },
                    Branching::VSIDS => {
                        let solver = Solver::<VSIDS, false>::new(graph, state, component_extractor, Box::<VSIDS>::default(), propagator, mlimit, $e);
                        GenericSolver::QVSIDS(solver)
                    },
                }
            }
        }
    };
}

macro_rules! search {
    ($s:expr) => {
        match $s {
            GenericSolver::SMinInDegree(mut solver) => solver.search(),
            GenericSolver::SMinOutDegree(mut solver) => solver.search(),
            GenericSolver::SMaxDegree(mut solver) => solver.search(),
            GenericSolver::SVSIDS(mut solver) => solver.search(),
            GenericSolver::QMinInDegree(mut solver) => solver.search(),
            GenericSolver::QMinOutDegree(mut solver) => solver.search(),
            GenericSolver::QMaxDegree(mut solver) => solver.search(),
            GenericSolver::QVSIDS(mut solver) => solver.search(),
        }
    }
}

macro_rules! compile {
    ($c:expr) => {
        match $c {
            GenericSolver::SMinInDegree(mut solver) => solver.compile(),
            GenericSolver::SMinOutDegree(mut solver) => solver.compile(),
            GenericSolver::SMaxDegree(mut solver) => solver.compile(),
            GenericSolver::SVSIDS(mut solver) => solver.compile(),
            GenericSolver::QMinInDegree(mut solver) => solver.compile(),
            GenericSolver::QMinOutDegree(mut solver) => solver.compile(),
            GenericSolver::QMaxDegree(mut solver) => solver.compile(),
            GenericSolver::QVSIDS(mut solver) => solver.compile(),
        }
    }
}

pub(crate) use make_solver;
pub(crate) use compile;
pub(crate) use search;
