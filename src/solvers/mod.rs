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

use crate::core::graph::{DistributionIndex, Graph};
use crate::branching::*;
use crate::core::components::ComponentExtractor;
use crate::propagator::Propagator;
use crate::Branching;

use search_trail::StateManager;
use rug::Float;
use std::hash::Hash;
use bitvec::prelude::*;

#[derive(Debug)]
pub enum Error {
    Unsat,
    Timeout,
}

pub type Bounds = (Float, Float);
/// Type alias used for the solution of the problem, which is either a Float or UNSAT
pub type ProblemSolution = Result<Bounds, Error>;

pub mod solver;
mod statistics;

pub use solver::Solver;

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

pub fn generic_solver(graph: Graph, state: StateManager, component_extractor: ComponentExtractor, branching: Branching, propagator: Propagator, mlimit: u64, epsilon: f64, timeout: u64, stat: bool) -> GenericSolver {
    if stat {
        match branching {
            Branching::MinInDegree => {
                let solver = Solver::<MinInDegree, true>::new(graph, state, component_extractor, Box::<MinInDegree>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::SMinInDegree(solver)
            },
            Branching::MinOutDegree => {
                let solver = Solver::<MinOutDegree, true>::new(graph, state, component_extractor, Box::<MinOutDegree>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::SMinOutDegree(solver)
            },
            Branching::MaxDegree => {
                let solver = Solver::<MaxDegree, true>::new(graph, state, component_extractor, Box::<MaxDegree>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::SMaxDegree(solver)
            },
            Branching::VSIDS => {
                let solver = Solver::<VSIDS, true>::new(graph, state, component_extractor, Box::<VSIDS>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::SVSIDS(solver)
            },
        }
    } else {
        match branching {
            Branching::MinInDegree => {
                let solver = Solver::<MinInDegree, false>::new(graph, state, component_extractor, Box::<MinInDegree>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::QMinInDegree(solver)
            },
            Branching::MinOutDegree => {
                let solver = Solver::<MinOutDegree, false>::new(graph, state, component_extractor, Box::<MinOutDegree>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::QMinOutDegree(solver)
            },
            Branching::MaxDegree => {
                let solver = Solver::<MaxDegree, false>::new(graph, state, component_extractor, Box::<MaxDegree>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::QMaxDegree(solver)
            },
            Branching::VSIDS => {
                let solver = Solver::<VSIDS, false>::new(graph, state, component_extractor, Box::<VSIDS>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::QVSIDS(solver)
            },
        }
    }
}

macro_rules! solver_from_problem {
    ($d:expr, $c:expr, $b:expr, $e:expr, $m:expr, $t:expr, $s:expr) => {
        {
            let mut state = StateManager::default();
            let graph = graph_from_problem($d, $c, &mut state);
            let propagator = Propagator::new(&mut state);
            let component_extractor = ComponentExtractor::new(&graph, &mut state);
            let mlimit = if let Some(m) = $m {
                m
            } else {
                u64::MAX
            };
            generic_solver(graph, state, component_extractor, $b, propagator, mlimit, $e, $t, $s)
        }
    };
}

macro_rules! make_solver {
    ($i:expr, $b:expr, $e:expr, $m:expr, $t:expr, $s:expr, $l:expr) => {
        {
            let mut state = StateManager::default();
            let propagator = Propagator::new(&mut state);
            let graph = graph_from_ppidimacs($i, &mut state, $l);
            let component_extractor = ComponentExtractor::new(&graph, &mut state);
            let mlimit = if let Some(m) = $m {
                m
            } else {
                u64::MAX
            };
            generic_solver(graph, state, component_extractor, $b, propagator, mlimit, $e, $t, $s)
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

macro_rules! lds {
    ($s:expr) => {
        match $s {
            GenericSolver::SMinInDegree(mut solver) => solver.lds(),
            GenericSolver::SMinOutDegree(mut solver) => solver.lds(),
            GenericSolver::SMaxDegree(mut solver) => solver.lds(),
            GenericSolver::SVSIDS(mut solver) => solver.lds(),
            GenericSolver::QMinInDegree(mut solver) => solver.lds(),
            GenericSolver::QMinOutDegree(mut solver) => solver.lds(),
            GenericSolver::QMaxDegree(mut solver) => solver.lds(),
            GenericSolver::QVSIDS(mut solver) => solver.lds(),
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

pub(crate) use solver_from_problem;
pub(crate) use make_solver;
pub(crate) use compile;
pub(crate) use search;
pub(crate) use lds;

/// A key of the cache. It is composed of
///     1. A hash representing the sub-problem being solved
///     2. The bitwise representation of the sub-problem being solved
/// 
/// We adopt this two-level representation for the cache key for efficiency reason. The hash is computed during
/// the detection of the components and is a XOR of random bit string. This is efficient but do not ensure that
/// two different sub-problems have different hash.
/// Hence, we also provide an unique representation of the sub-problem, using 64 bits words, in case of hash collision.
#[derive(Default)]
pub struct CacheKey {
    hash: u64,
    repr: BitVec,
}

impl CacheKey {
    pub fn new(hash: u64, repr: BitVec) -> Self {
        Self {
            hash,
            repr
        }
    }
}

impl Hash for CacheKey {

    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }

}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            false
        } else {
            self.repr == other.repr
        }
    }
}

impl Eq for CacheKey {}

/// An entry in the cache for the search. It contains the bounds computed when the sub-problem was
/// explored as well as various informations used by the solvers.
#[derive(Clone)]
pub struct SearchCacheEntry {
    /// The current bounds on the sub-problem
    bounds: Bounds,
    /// Maximum discrepancy used for that node
    discrepancy: usize,
    /// The distribution on which to branch in this problem
    distribution: Option<DistributionIndex>,
}

impl SearchCacheEntry {

    /// Returns a new cache entry
    pub fn new(bounds: Bounds, discrepancy: usize, distribution: Option<DistributionIndex>) -> Self {
        Self {
            bounds,
            discrepancy,
            distribution,
        }
    }

    /// Returns a reference to the bounds of this entry
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    /// Returns the discrepancy of the node
    pub fn discrepancy(&self) -> usize {
        self.discrepancy
    }

    pub fn distribution(&self) -> Option<DistributionIndex> {
        self.distribution
    }
}
