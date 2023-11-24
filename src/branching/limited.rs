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

use rustc_hash::FxHashSet;
use search_trail::{StateManager, ReversibleUsize, UsizeManager};
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::{DistributionIndex, Graph};
use crate::branching::BranchingDecision;


/// This heuristic selects the clause with the minimum in degree. In case of tie, it selects the clause
/// for which the less number of parents have been removed.
/// Then, it selects the first unfixed distribution from the clause.
pub struct Counting{
    /// The last distribution index (excluded)
    stop_i: usize,
    /// The current distribution index
    current_i: ReversibleUsize,
    /// The constrained distributions
    constrained_distributions: Vec<DistributionIndex>,
}

impl Counting {
    pub fn new(start_i: usize, len: usize, state:&mut StateManager, constrained_distributions: Vec<DistributionIndex>) -> Self {
        Self {
            current_i: state.manage_usize(start_i),
            stop_i: start_i + len,
            constrained_distributions,
        }
    }
}

impl BranchingDecision for Counting {
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &mut StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let mut curr = state.get_usize(self.current_i);
        let distrib_set = component_extractor.component_distribution_iter(component).collect::<FxHashSet<DistributionIndex>>();
        while curr < self.stop_i {
            let distrib = self.constrained_distributions[curr];
            if distrib_set.contains(&distrib) && g[distrib].is_constrained(state) {
                state.set_usize(self.current_i, curr + 1);
                return Some(distrib);
            }
            curr += 1;
        }
        None
    }
    
    fn init(&mut self, _g: &Graph, _state: &StateManager) {}
    fn update_distribution_score(&mut self, _distribution: DistributionIndex){}
    fn decay_scores(&mut self){}
}
