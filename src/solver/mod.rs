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

#![allow(dead_code)]
mod branching;
mod propagator;

use crate::core::graph::*;
use crate::core::trail::*;
use crate::solver::branching::BranchingHeuristic;
use crate::solver::propagator::SimplePropagator;

pub fn solve<B: BranchingHeuristic>(graph: &mut Graph, branching_heuristic: &mut B) -> f64 {
    if let Some(d) = branching_heuristic.branching_decision(graph) {
        let mut obj = 0.0;
        for node in d {
            graph.state.save_state();
            graph.propagate_node(node, true);
            obj += solve(graph, branching_heuristic);
            graph.state.restore_state();
        }
        obj
    } else {
        graph.get_objective()
    }
}
