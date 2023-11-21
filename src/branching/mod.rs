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
use crate::core::graph::{ DistributionIndex, Graph};

pub trait BranchingDecision {
    fn branch_on(&mut self, g: &Graph, state: &StateManager, component_extractor: &ComponentExtractor, component: ComponentIndex) -> Option<DistributionIndex>;
    fn init(&mut self, g: &Graph, state: &StateManager);
    fn update_distribution_score(&mut self, distribution: DistributionIndex);
    fn decay_scores(&mut self);
}

mod cgraph;
mod vsids;
mod limited;

pub use cgraph::{MaxDegree, MinInDegree, MinOutDegree};
pub use vsids::VSIDS;
pub use limited::Counting;
