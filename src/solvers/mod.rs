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
use crate::diagrams::dac::dac::{Dac, NodeIndex};
use crate::core::graph::{VariableIndex, DistributionIndex};


/// Unit structure representing the the problem is UNSAT
#[derive(Debug)]
pub struct Unsat;

/// Type alias used for the solution of the problem, which is either a Float or UNSAT
pub type ProblemSolution = Result<Float, Unsat>;

pub type Bounds = (Float, Float);

pub mod search;
pub mod compiler;
mod statistics;

pub use search::SearchSolver;
pub use compiler::DACCompiler;
pub use sysinfo::{SystemExt, System};

pub enum Solver {
    SMinInDegree(SearchSolver<MinInDegree, true>),
    SMinOutDegree(SearchSolver<MinOutDegree, true>),
    SMaxDegree(SearchSolver<MaxDegree, true>),
    SVSIDS(SearchSolver<VSIDS, true>),
    QMinInDegree(SearchSolver<MinInDegree, false>),
    QMinOutDegree(SearchSolver<MinOutDegree, false>),
    QMaxDegree(SearchSolver<MaxDegree, false>),
    QVSIDS(SearchSolver<VSIDS, false>),
}

impl Solver {

    pub fn solve(&mut self) -> ProblemSolution {
        match self {
            Solver::SMinInDegree(solver) => solver.solve(),
            Solver::SMinOutDegree(solver) => solver.solve(),
            Solver::SMaxDegree(solver) => solver.solve(),
            Solver::SVSIDS(solver) => solver.solve(),
            Solver::QMinInDegree(solver) => solver.solve(),
            Solver::QMinOutDegree(solver) => solver.solve(),
            Solver::QMaxDegree(solver) => solver.solve(),
            Solver::QVSIDS(solver) => solver.solve(),
        }
    }

    pub fn solve_partial(&mut self, propagations:&Vec<(VariableIndex, bool)>) -> f64 {
        match self {
            Solver::SMinInDegree(ref mut solver) => {
                solver.add_to_propagation_stack(propagations);
                match solver.solve() {
                    Ok(p) => {
                        p.to_f64()
                    },
                    Err(_) => {0.0}, // TODO what do we do if we can not evaluate the node ?
                }
            },
            Solver::SMinOutDegree(ref mut solver) => {
                solver.add_to_propagation_stack(propagations);
                match solver.solve() {
                    Ok(p) => {
                        p.to_f64()
                    },
                    Err(_) => {0.0}, // TODO what do we do if we can not evaluate the node ?
                }
            },
            Solver::SMaxDegree(ref mut solver) => {
                solver.add_to_propagation_stack(propagations);
                match solver.solve() {
                    Ok(p) => {
                        p.to_f64()
                    },
                    Err(_) => {0.0}, // TODO what do we do if we can not evaluate the node ?
                }
            },
            Solver::SVSIDS(ref mut solver) => {
                solver.add_to_propagation_stack(propagations);
                match solver.solve() {
                    Ok(p) => {
                        p.to_f64()
                    },
                    Err(_) => {0.0}, // TODO what do we do if we can not evaluate the node ?
                }
            },
            Solver::QMinInDegree(ref mut solver) => {
                solver.add_to_propagation_stack(propagations);
                match solver.solve() {
                    Ok(p) => {
                        p.to_f64()
                    },
                    Err(_) => {0.0}, // TODO what do we do if we can not evaluate the node ?
                }
            },
            Solver::QMinOutDegree(ref mut solver) => {
                solver.add_to_propagation_stack(propagations);
                match solver.solve() {
                    Ok(p) => {
                        p.to_f64()
                    },
                    Err(_) => {0.0}, // TODO what do we do if we can not evaluate the node ?
                }
            },
            Solver::QMaxDegree(ref mut solver) => {
                solver.add_to_propagation_stack(propagations);
                match solver.solve() {
                    Ok(p) => {
                        p.to_f64()
                    },
                    Err(_) => {0.0}, // TODO what do we do if we can not evaluate the node ?
                }
            },
            Solver::QVSIDS(ref mut solver) => {
                solver.add_to_propagation_stack(propagations);
                match solver.solve() {
                    Ok(p) => {
                        p.to_f64()
                    },
                    Err(_) => {0.0}, // TODO what do we do if we can not evaluate the node ?
                }
            },
        }
    }

    pub fn update_distributions(&mut self, distributions: &Vec<Vec<f64>>) {
        match self {
            Solver::SMinInDegree(ref mut solver) => solver.update_distributions(distributions),
            Solver::SMinOutDegree(ref mut solver) => solver.update_distributions(distributions),
            Solver::SMaxDegree(ref mut solver) => solver.update_distributions(distributions),
            Solver::SVSIDS(ref mut solver) => solver.update_distributions(distributions),
            Solver::QMinInDegree(ref mut solver) => solver.update_distributions(distributions),
            Solver::QMinOutDegree(ref mut solver) => solver.update_distributions(distributions),
            Solver::QMaxDegree(ref mut solver) => solver.update_distributions(distributions),
            Solver::QVSIDS(ref mut solver) => solver.update_distributions(distributions),
        }
    }

    pub fn reset_cache(&mut self) {
        match self {
            Solver::SMinInDegree(ref mut solver) => solver.reset_cache(),
            Solver::SMinOutDegree(ref mut solver) => solver.reset_cache(),
            Solver::SMaxDegree(ref mut solver) => solver.reset_cache(),
            Solver::SVSIDS(ref mut solver) => solver.reset_cache(),
            Solver::QMinInDegree(ref mut solver) => solver.reset_cache(),
            Solver::QMinOutDegree(ref mut solver) => solver.reset_cache(),
            Solver::QMaxDegree(ref mut solver) => solver.reset_cache(),
            Solver::QVSIDS(ref mut solver) => solver.reset_cache(),
        }
    }
}

macro_rules! make_solver {
    ($i:expr, $b:expr, $e:expr, $m:expr, $s:expr) => {
        {
            let mut state = StateManager::default();
            let propagator = Propagator::new(&mut state);
            let graph = graph_from_ppidimacs($i, &mut state);
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
                        let mut solver = SearchSolver::<MinInDegree, true>::new(graph, state, component_extractor, Box::<MinInDegree>::default(), propagator, mlimit, $e);
                        solver.init();
                        Solver::SMinInDegree(solver)
                    },
                    Branching::MinOutDegree => {
                        let mut solver = SearchSolver::<MinOutDegree, true>::new(graph, state, component_extractor, Box::<MinOutDegree>::default(), propagator, mlimit, $e);
                        solver.init();
                        Solver::SMinOutDegree(solver)
                    },
                    Branching::MaxDegree => {
                        let mut solver = SearchSolver::<MaxDegree, true>::new(graph, state, component_extractor, Box::<MaxDegree>::default(), propagator, mlimit, $e);
                        solver.init();
                        Solver::SMaxDegree(solver)
                    },
                    Branching::VSIDS => {
                        let mut solver = SearchSolver::<VSIDS, true>::new(graph, state, component_extractor, Box::<VSIDS>::default(), propagator, mlimit, $e);
                        solver.init();
                        Solver::SVSIDS(solver)
                    },
                }
            } else {
                match $b {
                    Branching::MinInDegree => {
                        let mut solver = SearchSolver::<MinInDegree, false>::new(graph, state, component_extractor, Box::<MinInDegree>::default(), propagator, mlimit, $e);
                        solver.init();
                        Solver::QMinInDegree(solver)
                    },
                    Branching::MinOutDegree => {
                        let mut solver = SearchSolver::<MinOutDegree, false>::new(graph, state, component_extractor, Box::<MinOutDegree>::default(), propagator, mlimit, $e);
                        solver.init();
                        Solver::QMinOutDegree(solver)
                    },
                    Branching::MaxDegree => {
                        let mut solver = SearchSolver::<MaxDegree, false>::new(graph, state, component_extractor, Box::<MaxDegree>::default(), propagator, mlimit, $e);
                        solver.init();
                        Solver::QMaxDegree(solver)
                    },
                    Branching::VSIDS => {
                        let mut solver = SearchSolver::<VSIDS, false>::new(graph, state, component_extractor, Box::<VSIDS>::default(), propagator, mlimit, $e);
                        solver.init();
                        Solver::QVSIDS(solver)
                    },
                }
            }
        }
    };
}

macro_rules! solve_search {
    ($s:expr) => {
        match $s {
            Solver::SMinInDegree(mut solver) => {
                solver.init();
                solver.solve()
            },
            Solver::SMinOutDegree(mut solver) => {
                solver.init();
                solver.solve()
            },
            Solver::SMaxDegree(mut solver) => {
                solver.init();
                solver.solve()
            },
            Solver::SVSIDS(mut solver) => {
                solver.init();
                solver.solve()
            },
            Solver::QMinInDegree(mut solver) => {
                solver.init();
                solver.solve()
            },
            Solver::QMinOutDegree(mut solver) => {
                solver.init();
                solver.solve()
            },
            Solver::QMaxDegree(mut solver) => {
                solver.init();
                solver.solve()
            },
            Solver::QVSIDS(mut solver) => {
                solver.init();
                solver.solve()
            },
        }
    }
}

pub enum Compiler {
    MinInDegree(DACCompiler<MinInDegree>),
    MinOutDegree(DACCompiler<MinOutDegree>),
    MaxDegree(DACCompiler<MaxDegree>),
    VSIDS(DACCompiler<VSIDS>),
}

impl Compiler {
    pub fn extend_partial_node_with(&mut self, node: NodeIndex, dac: &mut Dac, distribution: DistributionIndex) {
        match self {
            Compiler::VSIDS(ref mut compiler) => compiler.extend_partial_node_with(node, dac, distribution),
            Compiler::MinInDegree(ref mut compiler) => compiler.extend_partial_node_with(node, dac, distribution),
            Compiler::MinOutDegree(ref mut compiler) => compiler.extend_partial_node_with(node, dac, distribution),
            Compiler::MaxDegree(ref mut compiler) => compiler.extend_partial_node_with(node, dac, distribution),
        }
    }

}

macro_rules! make_compiler {
    ($i:expr, $b:expr, $r:expr) => {
        {
            let mut state = StateManager::default();
            let propagator = Propagator::new(&mut state);
            let graph = graph_from_ppidimacs($i, &mut state);
            let component_extractor = ComponentExtractor::new(&graph, &mut state);
            match $b {
                Branching::MinInDegree => {
                    Compiler::MinInDegree(DACCompiler::<MinInDegree>::new(graph, state, component_extractor, Box::<MinInDegree>::default(), propagator, $r))
                },
                Branching::MinOutDegree => {
                    Compiler::MinOutDegree(DACCompiler::<MinOutDegree>::new(graph, state, component_extractor, Box::<MinOutDegree>::default(), propagator, $r))
                },
                Branching::MaxDegree => {
                    Compiler::MaxDegree(DACCompiler::<MaxDegree>::new(graph, state, component_extractor, Box::<MaxDegree>::default(), propagator, $r))
                },
                Branching::VSIDS => {
                    Compiler::VSIDS(DACCompiler::<VSIDS>::new(graph, state, component_extractor, Box::<VSIDS>::default(), propagator, $r))
                },
            }
        }
    };
}

macro_rules! compile {
    ($c:expr) => {
        match $c {
            Compiler::MinInDegree(ref mut c) => c.compile(),
            Compiler::MinOutDegree(ref mut c) => c.compile(),
            Compiler::MaxDegree(ref mut c) => c.compile(),
            Compiler::VSIDS(ref mut c) => c.compile(),
        }
    }
}

pub(crate) use make_solver;
pub(crate) use make_compiler;
pub(crate) use compile;
pub(crate) use solve_search;
