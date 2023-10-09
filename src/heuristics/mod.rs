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

use search_trail::StateManager;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::{DistributionIndex, Graph};

/// Trait that defined the methods that a branching decision structure must implement.
pub trait BranchingDecision {
    /// Chooses one distribution from the component to branch on and returns it. If no distribution is present in
    /// the component, returns None.
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex>;
    
    /// Initialize, if necessary, the data structures used by the branching heuristics
    fn init(&mut self, g: &Graph, state: &StateManager);
}

pub mod branching_exact;