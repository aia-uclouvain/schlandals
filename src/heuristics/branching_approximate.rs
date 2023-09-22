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
use crate::heuristics::BranchingDecision;


#[derive(Default)]
pub struct MaxProbability;

impl BranchingDecision for MaxProbability {
    fn branch_on(
        &mut self,
        g: &Graph,
        state: &StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let mut best_score = usize::MAX;
        let mut best_distribution: Option<DistributionIndex> = None;
        let mut best_tie = 0.0;
        for clause in component_extractor.component_iter(component) {
            if g.is_clause_constrained(clause, state) && g.clause_has_probabilistic(clause, state) {                
                let score = g.get_clause_number_parents(clause, state);
                let (d, proba) = g.get_clause_active_distribution_highest_value(clause, state).unwrap();
                if score < best_score || (score == best_score && proba > best_tie) {
                    best_score = score;
                    best_tie = proba;
                    best_distribution = Some(d);
                }
            }
        }
        best_distribution
    }
    
    fn init(&mut self, _g: &Graph, _state: &StateManager) {}
    
}