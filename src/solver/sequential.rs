//Schlandalhttps://www.youtube.com/watch?v=-9lrYoX2cMks
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


//! This module provides the main structure of the solver. It is responsible for orchestring
//! all the different parts and glue them together.
//! The algorithm starts by doing an initial propagation of the variables indentified during the
//! parsing of the input file, and then solve recursively the problem.
//! It uses the branching decision to select which variable should be propagated next, call the propagator
//! and identified the independent component.
//! It is also responsible for updating the cache and clearing it when the memory limit is reached.
//! Finally it save and restore the states of the reversible variables used in the solver.

use std::hash::Hash;

use rustc_hash::FxHashMap;
use search_trail::{StateManager, SaveAndRestore};

use crate::common::f128;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::*;
use crate::solver::branching::BranchingDecision;
use crate::solver::propagator::FTReachablePropagator;
use crate::solver::statistics::Statistics;
use crate::common::PEAK_ALLOC;

use rug::Float;

/// Unit structure representing the the problem is UNSAT
#[derive(Debug)]
pub struct Unsat;

/// Type alias used for the solution of the problem, which is either a Float or UNSAT
type ProblemSolution = Result<Float, Unsat>;

/// The solver for a particular set of Horn clauses. It is generic over the branching heuristic
/// and has a constant parameter that tells if statistics must be recorded or not.
pub struct Solver<'b, B, const S: bool>
where
    B: BranchingDecision + ?Sized,
{
    /// Implication graph of the input CNF formula
    graph: Graph,
    /// State manager that allows to retrieve previous values when backtracking in the search tree
    state: StateManager,
    /// Extracts the connected components in the graph
    component_extractor: ComponentExtractor,
    /// Heuristics that decide on which distribution to branch next
    branching_heuristic: &'b mut B,
    /// The propagator
    propagator: FTReachablePropagator,
    /// Cache used to store results of sub-problems
    cache: FxHashMap<CacheEntry, Float>,
    /// Statistics collectors
    statistics: Statistics<S>,
    /// Memory limit allowed for the solver. This is a global memory limit, not a cache-size limit
    mlimit: u64,
}

/// A key of the cache. It is composed of
///     1. A hash representing the sub-problem being solved
///     2. The bitwise representation of the sub-problem being solved
/// 
/// We adopt this two-level representation for the cache key for efficiency reason. The hash is computed during
/// the detection of the components and is a XOR of random bit string. This is efficient but do not ensure that
/// two different sub-problems have different hash.
/// Hence, we also provide an unique representation of the sub-problem, using 64 bits words, in case of hash collision.
#[derive(Default)]
pub struct CacheEntry {
    hash: u64,
    repr: Vec<u64>,
}

impl CacheEntry {
    pub fn new(hash: u64, repr: Vec<u64>) -> Self {
        Self {
            hash,
            repr
        }
    }
}

impl Hash for CacheEntry {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl PartialEq for CacheEntry {
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            false
        } else {
            self.repr == other.repr
        }
    }
}

impl Eq for CacheEntry {}

impl<'b, B, const S: bool> Solver<'b, B, S>
where
    B: BranchingDecision + ?Sized,
{
    pub fn new(
        graph: Graph,
        state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: &'b mut B,
        propagator: FTReachablePropagator,
        mlimit: u64,
    ) -> Self {
        let cache = FxHashMap::default();
        Self {
            graph,
            state,
            component_extractor,
            branching_heuristic,
            propagator,
            cache,
            statistics: Statistics::default(),
            mlimit,
        }
    }

    /// Returns the solution for the sub-problem identified by the component. If the solution is in
    /// the cache, it is not computed. Otherwise it is solved and stored in the cache.
    fn get_cached_component_or_compute(&mut self, component: ComponentIndex) -> Float {
        self.statistics.cache_access();
        let bit_repr = self.graph.get_bit_representation(&self.state, component, &self.component_extractor);
        match self.cache.get(&bit_repr) {
            None => {
                self.statistics.cache_miss();
                let f = self.choose_and_branch(component);
                self.cache.insert(bit_repr, f.clone());
                f
            },
            Some(f) => {
                f.clone()
            }
        }
    }

    /// Chooses a distribution to branch on using the heuristics of the solver and returns the
    /// solution of the component.
    /// The solution is the sum of the probability of the SAT children.
    fn choose_and_branch(&mut self, component: ComponentIndex) -> Float {
        let decision = self.branching_heuristic.branch_on(
            &self.graph,
            &self.state,
            &self.component_extractor,
            component,
        );
        if let Some(distribution) = decision {
            self.statistics.or_node();
            let mut node_sol = f128!(0.0);
            for variable in self.graph.distribution_variable_iter(distribution) {
                self.state.save_state();
                match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, component, &self.component_extractor) {
                    Err(_) => {
                    }
                    Ok(v) => {
                        if v != 0.0 {
                            let mut child_sol = self._solve(component);
                            child_sol *= v;
                            node_sol += &child_sol;
                        }
                    }
                };
                self.state.restore_state();
            }
            node_sol
        } else {
            // The sub-formula is SAT, by definition return 1
            f128!(1.0)
        }
    }

    /// Solves the problem for the sub-graph identified by component.
    pub fn _solve(&mut self, component: ComponentIndex) -> Float {
        // If the memory limit is reached, clear the cache.
        if PEAK_ALLOC.current_usage_as_mb() as u64 >= self.mlimit {
            self.cache.clear();
        }
        self.state.save_state();
        // Default solution with a probability/count of 1
        // Since the probability are multiplied between the sub-components, it is neutral. And if
        // there are no sub-components, this is the default solution.
        let mut solution = f128!(1.0);
        // First we detect the sub-components in the graph
        if self
            .component_extractor
            .detect_components(&mut self.graph, &mut self.state, component, &mut self.propagator)
        {
            self.statistics.and_node();
            self.statistics
                .decomposition(self.component_extractor.number_components(&self.state));
            for sub_component in self.component_extractor.components_iter(&self.state) {
                solution *= self.get_cached_component_or_compute(sub_component);
                // If one sub-component has a probability of 0, then the probability of all sibling sub-components
                // are meaningless
                if solution == 0.0 {
                    break;
                }
            }
        }
        self.state.restore_state();
        solution
    }

    /// Solve the problems represented by the graph with the given branching heuristic.
    /// It finds all the assignments to the probabilistic variables for which there
    /// exists an assignment to the deterministic variables that respect the constraints.
    /// Each assignment is weighted by the product of the probabilistic variables assigned to true.
    pub fn solve(&mut self) -> ProblemSolution {
        // First set the number of clause in the propagator. This can not be done at the initialization of the propagator
        // because we need it to parse the input file as some variables might be detected as always being true or false.
        self.propagator.set_number_clauses(self.graph.number_clauses());
        // Doing an initial propagation to detect some UNSAT formula from the start
        match self.propagator.propagate(&mut self.graph, &mut self.state, ComponentIndex(0), &self.component_extractor) {
            Err(_) => ProblemSolution::Err(Unsat),
            Ok(p) => {
                // Checks if there are still constrained clauses in the graph
                let mut has_constrained = false;
                for clause in self.graph.clause_iter() {
                    if self.graph.is_clause_constrained(clause, &self.state) {
                        has_constrained = true;
                        break;
                    }
                }
                // If the graph still has constrained clauses, start the search.
                if has_constrained {
                    self.branching_heuristic.init(&self.graph, &self.state);
                    let mut solution = self._solve(ComponentIndex(0));
                    solution *= p;
                    self.statistics.print();
                    ProblemSolution::Ok(solution)
                } else {
                    ProblemSolution::Ok(p)
                }
            }
        }
    }
}