use search_trail::StateManager;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::problem::{DistributionIndex, Problem};

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
        let mut best_tie = usize::MAX;
        for clause in component_extractor.component_iter(component) {
            if g[clause].is_constrained(state) && !g[clause].is_learned() && g[clause].has_probabilistic(state) {
                let score = g[clause].number_constrained_children(state);
                let tie = g[clause].out_degree();
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
pub struct DLCS {}

impl BranchingDecision for DLCS {
    fn branch_on(
        &mut self,
        g: &Problem,
        state: &mut StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let mut distribution: Option<DistributionIndex> = None;
        let mut best_score = g.number_clauses();
        for d in component_extractor.component_distribution_iter(component) {
            let score = g[d].size(state);
            if score < best_score {
                best_score = score;
                distribution = Some(d);
            }
        }
        distribution
    }
    
    fn init(&mut self, _g: &Problem, _state: &StateManager) {}
}

#[derive(Default)]
pub struct DLCSVar {}

impl BranchingDecision for DLCSVar {
    fn branch_on(
        &mut self,
        g: &Problem,
        state: &mut StateManager,
        component_extractor: &ComponentExtractor,
        component: ComponentIndex,
    ) -> Option<DistributionIndex> {
        let mut distribution: Option<DistributionIndex> = None;
        let mut best_score = g.number_clauses();
        for d in component_extractor.component_distribution_iter(component) {
            let score = g[d].iter_variables().filter(|v| !g[*v].is_fixed(state)).map(|v| g[v].number_clauses(state)).max().unwrap();
            if score < best_score {
                best_score = score;
                distribution = Some(d);
            }
        }
        distribution
    }
    
    fn init(&mut self, _g: &Problem, _state: &StateManager) {}
}
