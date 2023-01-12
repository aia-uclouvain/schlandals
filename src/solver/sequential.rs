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

use crate::common::f128;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::*;
use crate::core::trail::*;
use crate::solver::branching::BranchingDecision;
use crate::solver::propagator::SimplePropagator;
use crate::solver::statistics::Statistics;
use rustc_hash::FxHashMap;

use rug::{Float, Integer};

use std::{fmt, ops};

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
#[derive(Clone)]
pub struct Solution {
    pub probability: Float,
    pub sol_count: Integer,
}

impl Solution {
    fn new(probability: Float, sol_count: Integer) -> Self {
        Solution {
            probability,
            sol_count,
        }
    }
}

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
    /// Cache used to store results of subtrees
    cache: FxHashMap<u64, Solution>,
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
    ) -> Self {
        Self {
            graph,
            state,
            component_extractor,
            branching_heuristic,
            cache: FxHashMap::default(),
            statistics: Statistics::default(),
        }
    }

    /// Returns the solution for the sub-graph identified by the component. If the solution is in
    /// the cache, it is not computed. Otherwise it is solved and stored in the cache.
    fn get_cached_component_or_compute(&mut self, component: ComponentIndex) -> &Solution {
        let hash = self.component_extractor.get_component_hash(component);
        // Need to rethink the hash strategy -> only the nodes is insufficient, need the edges
        self.statistics.cache_access();
        let should_compute = !self.cache.contains_key(&hash);
        if should_compute {
            let count = self.choose_and_branch(component);
            self.cache.insert(hash, count);
        } else {
            self.statistics.cache_hit();
        }
        self.component_extractor.decrement_cache_score(component);
        self.cache.get(&hash).unwrap()
    }

    /// Chooses a distribution to branch on using the heuristics of the solver and returns the
    /// solution of the component.
    /// The solution is the sum of the probability/count of the SAT children.
    fn choose_and_branch(&mut self, component: ComponentIndex) -> Solution {
        let decision = self.branching_heuristic.branch_on(
            &self.graph,
            &self.state,
            &self.component_extractor,
            component,
        );
        if let Some(distribution) = decision {
            self.statistics.or_node();
            let mut node_sol = Solution::new(f128!(0.0), Integer::new());
            for node in self.graph.distribution_iter(distribution) {
                self.state.save_state();
                match self.graph.propagate_node(node, true, &mut self.state) {
                    Err(_) => {}
                    Ok(v) => {
                        debug_assert_ne!(v, 0.0);
                        let mut child_sol = self._solve(component);
                        child_sol.probability *= v;
                        node_sol += &child_sol;
                    }
                };
                self.state.restore_state();
            }
            node_sol
        } else {
            // All distributions have been assigned 1 value, this is 1 assignement that is SAT for the probability variables
            // and extends to deterministic variables.
            let mut count = Integer::new();
            count += 1;
            Solution::new(f128!(1.0), count)
        }
    }

    /// Solves the problem for the sub-graph identified by component.
    pub fn _solve(&mut self, component: ComponentIndex) -> Solution {
        self.state.save_state();
        let mut count = Integer::with_capacity(self.graph.get_number_probabilistic());
        count += 1;
        // Default solution with a probability/count of 1
        // Since the probability are multiplied between the sub-components, it is neutral. And if
        // there are no sub-components, this is the default solution.
        let mut solution = Solution::new(f128!(1.0), count);
        // First we detect the sub-components in the graph
        if self
            .component_extractor
            .detect_components(&self.graph, &mut self.state, component)
        {
            if self.component_extractor.number_components(&self.state) != 0 {
                self.statistics.and_node();
                self.statistics
                    .decomposition(self.component_extractor.number_components(&self.state));
                for sub_component in self.component_extractor.components_iter(&self.state) {
                    if self.component_extractor.has_only_probabilistic(sub_component) {
                        let count = Integer::with_capacity(self.graph.get_number_probabilistic());
                        let mut solution = Solution::new(f128!(1.0), count);
                        for distribution in self
                            .component_extractor
                            .get_component_distributions(component)
                        {
                            let distribution_proba: f64 = self
                                .graph
                                .distribution_iter(*distribution)
                                .filter(|n| !self.graph.is_node_bound(*n, &self.state))
                                .map(|n| self.graph.get_node_weight(n).unwrap())
                                .sum();
                            solution.probability *= distribution_proba;
                        }
                    } else {
                        solution *= self.get_cached_component_or_compute(sub_component);
                    }
                    if solution.probability == 0.0 {
                        break;
                    }
                }
            }
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
    pub fn solve(&mut self) -> Solution {
        let s = self._solve(ComponentIndex(0));
        self.statistics.print();
        s
    }
}

impl ops::AddAssign<&Solution> for Solution {
    fn add_assign(&mut self, rhs: &Solution) {
        self.probability += &rhs.probability;
        self.sol_count += &rhs.sol_count;
    }
}

impl ops::MulAssign<&Solution> for Solution {
    fn mul_assign(&mut self, rhs: &Solution) {
        self.probability *= &rhs.probability;
        self.sol_count *= &rhs.sol_count;
    }
}

impl fmt::Display for Solution {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Node with probability {} and {} solutions",
            self.probability, self.sol_count
        )
    }
}