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
use crate::core::problem::{DistributionIndex, Problem};
use crate::core::problem::VariableIndex;

use arboretum_td::graph::HashMapGraph;
use arboretum_td::graph::MutableGraph;
use arboretum_td::solver::{AlgorithmTypes, AtomSolverType, Solver};
use arboretum_td::SafeSeparatorLimits;
use rustc_hash::FxHashSet;

pub trait BranchingDecision {
    fn branch_on(&mut self, g: &Problem, state: &mut StateManager, component_extractor: &ComponentExtractor, component: ComponentIndex) -> Option<DistributionIndex>;
    fn init(&mut self, g: &Problem, state: &StateManager);
}

/// This heuristic selects the clause with the minimum in degree. In case of tie, it selects the clause
/// for which the less number of parents have been removed.
/// Then, it selects the first unfixed distribution from the clause.
#[derive(Default)]
pub struct MinInDegree {}

impl BranchingDecision for MinInDegree {
    fn branch_on(
        &mut self,
        g: &Problem,
        state: &mut StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let mut selected : Option<DistributionIndex> = None;
        let mut best_score = usize::MAX;
        let mut best_tie = usize::MAX;
        for clause in component_extractor.component_iter(component) {
            if g[clause].is_constrained(state) && !g[clause].is_learned() && g[clause].has_probabilistic(state) {
                let score = g[clause].number_constrained_parents(state);
                let tie = g[clause].in_degree();
                if score < best_score || (score == best_score && tie < best_tie) {
                    if let Some(d) = g[clause].get_constrained_distribution(state, g) {
                        selected = Some(d);
                        best_score = score;
                        best_tie = tie;
                    }
                }
            }
        }
        selected
    }
    
    fn init(&mut self, _g: &Problem, _state: &StateManager) {}
    
}

#[derive(Default)]
pub struct MinOutDegree {}

impl BranchingDecision for MinOutDegree {
    fn branch_on(
        &mut self,
        g: &Problem,
        state: &mut StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let mut selected : Option<DistributionIndex> = None;
        let mut best_score = usize::MAX;
        for clause in component_extractor.component_iter(component) {
            if g[clause].is_constrained(state) && !g[clause].is_learned() && g[clause].has_probabilistic(state) {
                let score = g[clause].number_constrained_children(state);
                if score < best_score {
                    if let Some(d) = g[clause].get_constrained_distribution(state, g) {
                        selected = Some(d);
                        best_score = score;
                    }
                }
            }
        }
        selected
    }
    
    fn init(&mut self, _g: &Problem, _state: &StateManager) {}
    
}

#[derive(Default)]
pub struct TreeDecomposition {
    depth: Vec<usize>,
}

impl BranchingDecision for TreeDecomposition {

    fn branch_on(&mut self, g: &Problem, state: &mut StateManager, component_extractor: &ComponentExtractor, component: ComponentIndex) -> Option<DistributionIndex> {
        let mut selected : Option<DistributionIndex> = None;
        let mut best_score = usize::MAX;
        let mut best_tie = usize::MAX;
        for clause in component_extractor.component_iter(component) {
            if g[clause].is_constrained(state) && !g[clause].is_learned() && g[clause].has_probabilistic(state) {
                if let Some(distribution) = g[clause].get_constrained_distribution(state, g) {
                    let score = self.depth[distribution.0];
                    let tie = g[clause].number_constrained_parents(state);
                    if score < best_score || (score == best_score && tie < best_tie) {
                        selected = Some(distribution);
                        best_score = score;
                        best_tie = tie;
                    }
                }
            }
        }
        selected
    }

    fn init(&mut self, g: &Problem, _state: &StateManager) {
        println!("Initializing TD heuristic");
        self.depth = vec![usize::MAX; g.number_distributions()];
        let mut graph = HashMapGraph::new();
        for clause in g.clauses_iter() {
            let variables = g[clause].iter().map(|l| l.to_variable()).collect::<Vec<VariableIndex>>();
            for i in 0..variables.len() {
                for j in i+1..variables.len() {
                    graph.add_edge(variables[i].0, variables[j].0);
                }
            }
            for distribution in g.distributions_iter() {
                let variables = g[distribution].iter_variables().collect::<Vec<VariableIndex>>();
                for i in 0..variables.len() {
                    for j in i+1..variables.len() {
                        graph.add_edge(variables[i].0, variables[j].0);
                    }
                }
            }
        }

        println!("Solving TD");
        arboretum_td::timeout::initialize_timeout(60);
        let td = Solver::default_heuristic()
            .safe_separator_limits(
                SafeSeparatorLimits::default().use_min_degree_for_minor_safe(true),
            )
            .algorithm_types(
                AlgorithmTypes::default().atom_solver(AtomSolverType::TabuLocalSearchInfinite),
            ).seed(None).solve(&graph);
        //let td = Solver::default_exact().seed(None).safe_separator_limits(SafeSeparatorLimits::default().use_min_degree_for_minor_safe(true)).solve(&graph);
        println!("TD solved");
        println!("TD root: {:?}", td.root);
        let mut visited = FxHashSet::<usize>::default();
        let mut depth = 0;
        for meta_node in td.dfs() {
            visited.insert(meta_node.id);
            for var_id in meta_node.vertex_set.iter().copied() {
                let variable = VariableIndex(var_id);
                if let Some(distribution) = g[variable].distribution() {
                    if depth < self.depth[distribution.0] {
                        self.depth[distribution.0] = depth;
                    }
                }
            }
            let mut must_backtrack = true;
            for next in meta_node.neighbors.iter() {
                if !visited.contains(next) {
                    must_backtrack = false;
                    break;
                }
            }
            if must_backtrack {
                depth -= 1;
            } else {
                depth += 1;
            }
        }
    }

}
