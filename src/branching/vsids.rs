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
use crate::core::graph::{ DistributionIndex, Graph};
use super::BranchingDecision;

pub struct VSIDS {
    scores: Vec<f64>,
    vsids_increment: f64,
    vsids_decay: f64,
}

impl Default for VSIDS {
    fn default() -> Self {
        Self {
            scores: vec![],
            vsids_increment: 1.0,
            vsids_decay: 0.75,
        }
    }
}

impl BranchingDecision for VSIDS {
    fn branch_on(&mut self, _g: &Graph, _state: &mut StateManager, component_extractor: &ComponentExtractor, component: ComponentIndex) -> Option<DistributionIndex> {
        let mut best: Option<DistributionIndex> = None;
        let mut best_value = 0.0;
        for distribution in component_extractor.component_distribution_iter(component) {
            let score = self.scores[distribution.0];
            if score > best_value {
                best = Some(distribution);
                best_value = score;
            }
        }
        best
    }

    fn init(&mut self, g: &Graph, _state: &StateManager) {
        self.scores.resize(g.variables_iter().filter(|v| g[*v].is_probabilitic()).count(), 0.0);
        for clause in g.clauses_iter() {
            for v in g[clause].iter_probabilistic_variables() {
                self.scores[v.0] += 1.0;
            }
        }
    }

    fn update_distribution_score(&mut self, distribution: DistributionIndex) {
        self.scores[distribution.0] += self.vsids_increment;
    }

    fn decay_scores(&mut self) {
        self.vsids_increment *= 1.0 / self.vsids_decay;

        if self.vsids_increment > 1e100 {
            for i in 0..self.scores.len() {
                self.scores[i] *= 1e-100;
            }
            self.vsids_increment *= 1e-100;
        }
    }
}
