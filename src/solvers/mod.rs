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

use crate::core::problem::{DistributionIndex, Problem};
use crate::branching::*;
use crate::core::components::ComponentExtractor;
use crate::propagator::Propagator;
use crate::Branching;
use crate::common::FLOAT_CMP_THRESHOLD;
use crate::core::bitvec::Bitvec;
use crate::diagrams::NodeIndex;

use search_trail::StateManager;
use rug::Float;
use std::hash::Hash;

pub type Bounds = (Float, Float);

/// This structure represent a (possibly partial) solution found by the solver.
/// It is represented by a lower- and upper-bound on the true probability at the time at which the
/// solution was found.
#[derive(Clone)]
pub struct Solution {
    /// Lower bound on the true probability
    lower_bound: Float,
    /// Upper bound on the true probability
    upper_bound: Float,
    /// Number of seconds, since the start of the search, at which the solution was found
    time_found: u64,
}

impl Solution {

    pub fn new(lower_bound: Float, upper_bound: Float, time_found: u64) -> Self {
        Self {
            lower_bound,
            upper_bound,
            time_found,
        }
    }

    pub fn has_converged(&self, epsilon: f64) -> bool {
        self.upper_bound <= self.lower_bound.clone()*(1.0 + epsilon).powf(2.0) + FLOAT_CMP_THRESHOLD
    }

    pub fn print(&self) {
        println!("{}", self);
    }

    pub fn to_f64(&self) -> f64 {
        (self.lower_bound.clone() * self.upper_bound.clone()).sqrt().to_f64()
    }

    pub fn bounds(&self) -> (f64, f64) {
        (self.lower_bound.to_f64(), self.upper_bound.to_f64())
    }
}

impl std::fmt::Display for Solution {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Bounds on the probability [{:.8}, {:.8}] found in {} seconds", self.lower_bound, self.upper_bound, self.time_found)
    }
}

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

pub fn generic_solver(problem: Problem, state: StateManager, component_extractor: ComponentExtractor, branching: Branching, propagator: Propagator, mlimit: u64, epsilon: f64, timeout: u64, stat: bool) -> GenericSolver {
    if stat {
        match branching {
            Branching::MinInDegree => {
                let solver = Solver::<MinInDegree, true>::new(problem, state, component_extractor, Box::<MinInDegree>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::SMinInDegree(solver)
            },
            Branching::MinOutDegree => {
                let solver = Solver::<MinOutDegree, true>::new(problem, state, component_extractor, Box::<MinOutDegree>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::SMinOutDegree(solver)
            },
            Branching::MaxDegree => {
                let solver = Solver::<MaxDegree, true>::new(problem, state, component_extractor, Box::<MaxDegree>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::SMaxDegree(solver)
            },
            Branching::VSIDS => {
                let solver = Solver::<VSIDS, true>::new(problem, state, component_extractor, Box::<VSIDS>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::SVSIDS(solver)
            },
        }
    } else {
        match branching {
            Branching::MinInDegree => {
                let solver = Solver::<MinInDegree, false>::new(problem, state, component_extractor, Box::<MinInDegree>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::QMinInDegree(solver)
            },
            Branching::MinOutDegree => {
                let solver = Solver::<MinOutDegree, false>::new(problem, state, component_extractor, Box::<MinOutDegree>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::QMinOutDegree(solver)
            },
            Branching::MaxDegree => {
                let solver = Solver::<MaxDegree, false>::new(problem, state, component_extractor, Box::<MaxDegree>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::QMaxDegree(solver)
            },
            Branching::VSIDS => {
                let solver = Solver::<VSIDS, false>::new(problem, state, component_extractor, Box::<VSIDS>::default(), propagator, mlimit, epsilon, timeout);
                GenericSolver::QVSIDS(solver)
            },
        }
    }
}

macro_rules! solver_from_problem {
    ($d:expr, $c:expr, $b:expr, $e:expr, $m:expr, $t:expr, $s:expr) => {
        {
            let mut state = StateManager::default();
            let problem = problem_from_problem($d, $c, &mut state);
            let propagator = Propagator::new(&mut state);
            let component_extractor = ComponentExtractor::new(&problem, &mut state);
            let mlimit = if let Some(m) = $m {
                m
            } else {
                u64::MAX
            };
            generic_solver(problem, state, component_extractor, $b, propagator, mlimit, $e, $t, $s)
        }
    };
}

macro_rules! make_solver {
    ($i:expr, $b:expr, $e:expr, $m:expr, $t: expr, $s:expr, $u: expr) => {
        {
            let mut state = StateManager::default();
            let propagator = Propagator::new(&mut state);
            let problem = problem_from_cnf($i, &mut state, false, $u);
            let component_extractor = ComponentExtractor::new(&problem, &mut state);
            let mlimit = if let Some(m) = $m {
                m
            } else {
                u64::MAX
            };
            generic_solver(problem, state, component_extractor, $b, propagator, mlimit, $e, $t, $s)
        }
    };
}

macro_rules! search {
    ($s:expr) => {
        match $s {
            GenericSolver::SMinInDegree(mut solver) => solver.search(false),
            GenericSolver::SMinOutDegree(mut solver) => solver.search(false),
            GenericSolver::SMaxDegree(mut solver) => solver.search(false),
            GenericSolver::SVSIDS(mut solver) => solver.search(false),
            GenericSolver::QMinInDegree(mut solver) => solver.search(false),
            GenericSolver::QMinOutDegree(mut solver) => solver.search(false),
            GenericSolver::QMaxDegree(mut solver) => solver.search(false),
            GenericSolver::QVSIDS(mut solver) => solver.search(false),
        }
    }
}

macro_rules! lds {
    ($s:expr) => {
        match $s {
            GenericSolver::SMinInDegree(mut solver) => solver.search(true),
            GenericSolver::SMinOutDegree(mut solver) => solver.search(true),
            GenericSolver::SMaxDegree(mut solver) => solver.search(true),
            GenericSolver::SVSIDS(mut solver) => solver.search(true),
            GenericSolver::QMinInDegree(mut solver) => solver.search(true),
            GenericSolver::QMinOutDegree(mut solver) => solver.search(true),
            GenericSolver::QMaxDegree(mut solver) => solver.search(true),
            GenericSolver::QVSIDS(mut solver) => solver.search(true),
        }
    }
}

macro_rules! compile {
    ($c:expr, $l:expr) => {
        match $c {
            GenericSolver::SMinInDegree(mut solver) => solver.compile($l),
            GenericSolver::SMinOutDegree(mut solver) => solver.compile($l),
            GenericSolver::SMaxDegree(mut solver) => solver.compile($l),
            GenericSolver::SVSIDS(mut solver) => solver.compile($l),
            GenericSolver::QMinInDegree(mut solver) => solver.compile($l),
            GenericSolver::QMinOutDegree(mut solver) => solver.compile($l),
            GenericSolver::QMaxDegree(mut solver) => solver.compile($l),
            GenericSolver::QVSIDS(mut solver) => solver.compile($l),
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
#[derive(Default, Clone)]
pub struct CacheKey {
    hash: u64,
    repr: Bitvec,
}

impl CacheKey {
    pub fn new(hash: u64, repr: Bitvec) -> Self {
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
    /// The cache keys of the subcomponents related to the variable of the distribution
    variable_component_keys: Vec<(usize, Vec<CacheKey>)>,
    /// The distribution variables (new, old) that were fixed by the propagation for each variable of the distribution (new, old)
    forced_distribution_variables: Vec<(usize, Vec<(DistributionIndex, DistributionIndex, usize, usize)>)>,
    /// The distribution variables (new, old) that were uncontrained by the propagation and not summing to 1 and their distribution (new, old)
    unconstrained_distribution_variables: Vec<(usize, Vec<(DistributionIndex, DistributionIndex, Vec<(usize, usize)>)>)>,
    /// The node index of the cache entry in the diagram if is has already been created
    node_index: Option<NodeIndex>,
}

impl SearchCacheEntry {

    /// Returns a new cache entry
    pub fn new(bounds: Bounds, discrepancy: usize, distribution: Option<DistributionIndex>, variable_component_keys: Vec<(usize, Vec<CacheKey>)>, 
               forced_distribution_variables: Vec<(usize, Vec<(DistributionIndex, DistributionIndex, usize, usize)>)>, unconstrained_distribution_variables: Vec<(usize, Vec<(DistributionIndex, DistributionIndex, Vec<(usize, usize)>)>)>) -> Self {
        Self {
            bounds,
            discrepancy,
            distribution,
            variable_component_keys,
            forced_distribution_variables,
            unconstrained_distribution_variables,
            node_index: None,
        }
    }

    /// Returns a reference to the bounds of this entry
    pub fn bounds(&self) -> Bounds {
        self.bounds.clone()
    }

    /// Returns the discrepancy of the node
    pub fn discrepancy(&self) -> usize {
        self.discrepancy
    }

    pub fn distribution(&self) -> Option<DistributionIndex> {
        self.distribution
    }

    pub fn variable_component_keys(&self) -> Vec<(usize, Vec<CacheKey>)> {
        self.variable_component_keys.clone()
    }

    pub fn forced_distribution_variables_of(&self, variable: usize) -> Option<Vec<(DistributionIndex, DistributionIndex, usize, usize)>> {
        self.forced_distribution_variables.iter().find(|(v, _)| *v == variable).map(|forced| forced.1.clone())
    }

    pub fn unconstrained_distribution_variables_of(&self, variable: usize) -> Option<Vec<(DistributionIndex, DistributionIndex, Vec<(usize, usize)>)>> {
        self.unconstrained_distribution_variables.iter().find(|(v, _)| *v == variable).map(|unconstr| unconstr.1.clone())
    }

    pub fn node_index(&self) -> Option<NodeIndex> {
        self.node_index
    }

    pub fn set_node_index(&mut self, node_index: NodeIndex) {
        self.node_index = Some(node_index);
    }
}
