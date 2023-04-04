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

use search_trail::{StateManager, SaveAndRestore};

use crate::common::f128;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::*;
use crate::solver::branching::BranchingDecision;
use crate::solver::propagator::FTReachablePropagator;
use crate::solver::statistics::Statistics;
use crate::solver::cache::Cache;

use rug::{Float};

/// A solution of a node in the search space. It gives its probability of the sub-graph induced in
/// the node and the number of solutions in its subtree.
/// The probability of a node can be computed as follows
///     - If there are no probabilistic node in the sub-graph of the node, then the probability is
///     1
///     - Otherwise take some distribution with nodes p1, ..., pn in the sub-graph. Let us denote
///     N1, ..., Nn the nodes obtained by assigning pi to true. Then the probability of the node is
///     P(N1) + ...  + P(Nn)
/// If the problem is divided in multiple components (there are no nodes in multiple components)
/// then the probabilities of the components are multiplied).
///
/// The counts can be computed in the same manner.
///     - If there are no probabilistic nodes in the sub-graph, then the solution count is 1 by
///     default
///     - If not, using the same procedure as above, the count is the sum of the count of the
///     children (multiplied for independent sub-components)


#[derive(Debug)]
pub struct Unsat;

type ProblemSolution = Result<Float, Unsat>;

pub struct Solver<'b, B, const S: bool>
where
    B: BranchingDecision + ?Sized,
{
    /// Graph representing the input formula
    graph: Graph,
    /// State manager that allows to retrieve previous values when backtracking in the search tree
    state: StateManager,
    /// Extracts the connected components in the graph
    component_extractor: ComponentExtractor,
    /// Heuristics that decide on which distribution to branch next
    branching_heuristic: &'b mut B,
    /// The propagator
    propagator: FTReachablePropagator,
    /// Cache used to store results of subtrees
    cache: Cache,
    /// Statistics collectors
    statistics: Statistics<S>,
}

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
        let cache = Cache::new(mlimit, &graph);
        Self {
            graph,
            state,
            component_extractor,
            branching_heuristic,
            propagator,
            cache,
            statistics: Statistics::default(),
        }
    }

    /// Returns the solution for the sub-graph identified by the component. If the solution is in
    /// the cache, it is not computed. Otherwise it is solved and stored in the cache.
    fn get_cached_component_or_compute(&mut self, component: ComponentIndex) -> Float {
        // Need to rethink the hash strategy -> only the nodes is insufficient, need the edges
        self.statistics.cache_access();
        let cache_entry = self.cache.get(&self.component_extractor, component, &self.graph, &self.state);
        match cache_entry.0 {
            Some(v) => v,
            None => {
                self.statistics.cache_miss();
                let v = self.choose_and_branch(component);
                let (hash, start) = cache_entry.1;
                self.cache.set(hash, start, v.clone());
                v
            }
        }
    }

    /// Chooses a distribution to branch on using the heuristics of the solver and returns the
    /// solution of the component.
    /// The solution is the sum of the probability/count of the SAT children.
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
                match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state) {
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
            // All distributions have been assigned 1 value, this is 1 assignement that is SAT for the probability variables
            // and extends to deterministic variables.
            f128!(1.0)
        }
    }

    /// Solves the problem for the sub-graph identified by component.
    pub fn _solve(&mut self, component: ComponentIndex) -> Float {
        self.state.save_state();
        // Default solution with a probability/count of 1
        // Since the probability are multiplied between the sub-components, it is neutral. And if
        // there are no sub-components, this is the default solution.
        let mut solution = f128!(1.0);
        // First we detect the sub-components in the graph
        if self
            .component_extractor
            .detect_components(&self.graph, &mut self.state, component, &mut self.propagator)
        {
            match self.propagator.propagate_unconstrained_clauses(&mut self.graph, &mut self.state) {
                Ok(v) => solution *= v,
                Err(_) => {
                    panic!("Propagating unconstrained clauses raised UNSAT, should not happen");
                }
            };
            self.statistics.and_node();
            self.statistics
                .decomposition(self.component_extractor.number_components(&self.state));
            for sub_component in self.component_extractor.components_iter(&self.state) {
                solution *= self.get_cached_component_or_compute(sub_component);
                if solution == 0.0 {
                    break;
                }
            }
        } else {
            match self.propagator.propagate_unconstrained_clauses(&mut self.graph, &mut self.state) {
                Ok(v) => solution *= v,
                Err(_) => {
                    panic!("Propagating unconstrained clauses raised UNSAT, should not happen");
                }
            };
        }
        self.state.restore_state();
        solution
    }

    /// Solve the problems represented by the graph with the given branching heuristic.
    /// This means that it find all the assignment to the probabilistic nodes for which there
    /// exists an assignment to the deterministic nodes that respect the constraints in the graph.
    /// Each assignment is weighted by the product (or sum in log-domain) of the probabilistic
    /// nodes assigned to true. The solution of the root node is the sum of the weights of such
    /// assigments.
    pub fn solve(&mut self) -> ProblemSolution {
        match self.propagator.propagate(&mut self.graph, &mut self.state) {
            Err(_) => ProblemSolution::Err(Unsat),
            Ok(p) => {
                let mut solution = self._solve(ComponentIndex(0));
                solution *= p;
                self.statistics.print();
                ProblemSolution::Ok(solution)
            }
        }
    }
}