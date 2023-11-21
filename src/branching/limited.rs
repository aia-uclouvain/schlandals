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

//! This module provides several branching heuristics for our solver.
//! The branching heuristics work a bit differently than in classical search-based solvers.
//! Remember that we are solving a _projected weighted model counting_ problems, in which the
//! probabilistic variables are the decision variables (on which the number of models must be counted).
//! In addition to that, we impose that the probabilistic variables are partitionned in distributions, in
//! which the variables are mutually exclusive.
//! This means that the branching decision is reduced to choosing a distribution and then assigning each of its
//! variable to true.
//! Hence, the heuristics provided here returns a distribution (if any are present in the component) instead of a
//! variable.

use search_trail::StateManager;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::{ClauseIndex, DistributionIndex, Graph};
use crate::branching::BranchingDecision;


/// This heuristic selects the clause with the minimum in degree. In case of tie, it selects the clause
/// for which the less number of parents have been removed.
/// Then, it selects the first unfixed distribution from the clause.
#[derive(Default)]
pub struct Counting{
    /// The starting distribution index
    start_i: usize,
    /// The number of distributions to branch on
    len: usize,
}

impl Counting {
    pub fn new(start_i: usize, len: usize) -> Self {
        Self {
            start_i,
            len,
        }
    }
}

impl BranchingDecision for Counting {
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let mut best_clause: Option<ClauseIndex> = None;
        let mut best_score = usize::MAX;
        let mut best_tie = usize::MAX;
        for clause in component_extractor.component_iter(component) {
            if g[clause].is_constrained(state) && !g[clause].is_learned() && g[clause].has_probabilistic(state) {
                let score = g[clause].number_constrained_parents(state);
                let tie = g[clause].number_parents();
                if score < best_score || (score == best_score && tie < best_tie) {
                    best_score = score;
                    best_tie = tie;
                    best_clause = Some(clause);
                }
            }
        }
        
        match best_clause {
            Some(clause) => {
                g[clause].get_constrained_distribution(state, g)
            },
            None => None
        }
    }
    
    fn init(&mut self, _g: &Graph, _state: &StateManager) {}
    fn update_distribution_score(&mut self, distribution: DistributionIndex){}
    fn decay_scores(&mut self){}
}
