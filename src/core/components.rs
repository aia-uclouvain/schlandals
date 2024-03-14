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

//! This module provides structure used to detect connected components in the implication graph.
//! Two clauses C1 = (I1, h1) and C2 = (I2, h2) are connected if and only if
//!     1. Both clauses are still constrained in the implication graph
//!     2. Either h2 is in I1 or h1 is in I2
//!     3. C1 and C2 have probabilistic variables in their bodies that are from the same distribution
//!   
//! The components are extracted using a simple DSF on the implication graph. During this traversal, the
//! hash of the components are also computing.
//! This hash is a simple XOR between random bitstring that are assigned durnig the graph creation.
//! For convenience the DFS extractor also sets the f-reachability and t-reachability once the component is extracted.
//! 
//! This module also exposes a special component extractor that do not detect any components.
//! It should only be used for debugging purpose to isolate bugs

use super::graph::{ClauseIndex, Graph, DistributionIndex};
use crate::{propagator::Propagator, solvers::CacheKey};
use search_trail::{ReversibleUsize, StateManager, UsizeManager};
use bitvec::prelude::*;

/// Abstraction used as a typesafe way of retrieving a `Component`
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ComponentIndex(pub usize);

/// This structure is an extractor of component that works by doing a DFS on the component to
/// extract sub-components. It keeps all the `ClauseIndex` in a single vector (the `clauses`
/// vector) and maintain the property that all the clauses of a component are in a contiguous
/// part of the vectors. The components can then be identified by two usize
///     1. The index of the first clause in the component
///     2. The size of the component
///
/// A second vector, `positions`, is used to keep track of the position of each clause in `clauses`
/// (since they are swapped from their initial position).
/// For instance let us assume that we have two components, the `clauses` vector might looks like
/// [0, 2, 4, 1 | 5, 7 ] (the | is the separation between the two components, and integers
/// represent the clauses).
/// If now the first component is split in two, the clauses are moved in the sub-vectors spanning the
/// first four indexes. Thus (assuming 0 and 1 are in the same component and 2 and 4 in another),
/// this is a valid representation of the new vector [0, 1 | 2, 4 | 5, 7] while this is not
/// [2, 4 | 5, 7 | 0, 1].
pub struct ComponentExtractor {
    /// The vector containing the clauses. All clauses in a component are in a contiguous part of this
    /// vector
    clauses: Vec<ClauseIndex>,
    /// The vector mapping for each `ClauseIndex` its position in `clauses`
    clause_positions: Vec<usize>,
    /// Holds the components computed by the extractor during the search
    components: Vec<Component>,
    /// The index of the first component of the current node in the search tree
    base: ReversibleUsize,
    /// The first index which is not a component of the current node in the search tree
    limit: ReversibleUsize,
    /// Which variable has been seen during the exploration, for the hash computation
    seen_var: Vec<bool>,
    /// Vector to store the distributions of each component. As for the clause, a distribution can be in only one component at a time
    distributions: Vec<DistributionIndex>,
    /// Vector to store the position of each distribution in the vector
    distribution_positions: Vec<usize>,
    /// Stack for clause to process during exploration
    exploration_stack: Vec<ClauseIndex>,
}

/// A Component is identified by two integers. The first is the index in the vector of clauses at
/// which the component starts, and the second is the size of the component.
#[derive(Debug, Clone)]
pub struct Component {
    /// First index of the component
    start: usize,
    /// Size of the component
    size: usize,
    /// First index of the distribution
    distribution_start: usize,
    /// Number of distribution in the component
    number_distribution: usize,
    /// Hash of the component, computed during its detection
    hash: u64,
    /// Maximum probability of the sub-problem represented by the component (i.e., all remaining
    /// valid interpretation are models)
    max_probability: f64,
    /// If the distributions are splitted into (non-)trainable, then this flags indicates if there
    /// are trainable distributions in the component.
    has_learned_distribution: bool,
    /// Bitwise representation of the problem
    bit_repr: BitVec,
}

impl Component {

    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn max_probability(&self) -> f64 {
        self.max_probability
    }

    pub fn has_learned_distribution(&self) -> bool {
        self.has_learned_distribution
    }

    pub fn get_cache_key(&self) -> CacheKey {
        CacheKey::new(self.hash, self.bit_repr.clone())
    }

    pub fn number_distribution(&self) -> usize {
        self.number_distribution
    }
}

impl ComponentExtractor {
    /// Creates a new component extractor for the implication graph `g`
    pub fn new(g: &Graph, state: &mut StateManager) -> Self {
        let nodes = (0..g.number_clauses()).map(ClauseIndex).collect();
        let clause_positions = (0..g.number_clauses()).collect();
        let distributions = (0..g.number_distributions()).map(DistributionIndex).collect();
        let distribution_positions = (0..g.number_distributions()).collect();
        let components = vec![Component {
            start: 0,
            size: g.number_clauses(),
            distribution_start: 0,
            number_distribution: g.number_distributions(),
            hash: 0,
            max_probability: 1.0,
            has_learned_distribution: false,
            bit_repr: bits![1].repeat(g.number_variables() + g.number_clauses_probem()),
        }];
        Self {
            clauses: nodes,
            clause_positions,
            components,
            base: state.manage_usize(0),
            limit: state.manage_usize(1),
            seen_var: vec![false; g.number_variables()],
            distributions,
            distribution_positions,
            exploration_stack: vec![],
        }
    }

    /// Returns true if and only if:
    ///     1. The clause is constrained
    ///     2. It is part of the current component being splitted in sub-components
    ///     3. It has not yet been added to the component
    fn is_node_visitable(
        &self,
        g: &Graph,
        clause: ClauseIndex,
        comp_start: usize,
        comp_size: &usize,
        state: &StateManager,
    ) -> bool {
        // if the clause has already been visited, then its position in the component must
        // be between [start..(start + size)].
        let clause_pos = self.clause_positions[clause.0];
        g[clause].is_constrained(state) && !(comp_start <= clause_pos && clause_pos < (comp_start + *comp_size))
    }
    
    /// Returns true if the distribution has not yet been visited during the component exploration
    fn is_distribution_visitable(&self, distribution: DistributionIndex, distribution_start: usize, distribution_size: &usize) -> bool {
        let distribution_pos = self.distribution_positions[distribution.0];
        !(distribution_start <= distribution_pos && distribution_pos < (distribution_start + *distribution_size))
    }

    /// Adds the clause to the component and recursively explores its children, parents and clauses sharing a distribution
    /// with it. If the clause is unconstrained or has already been visited, it is skipped.
    fn explore_component(
        &mut self,
        g: &Graph,
        comp_start: usize,
        comp_size: &mut usize,
        comp_distribution_start: usize,
        comp_number_distribution: &mut usize,
        hash: &mut u64,
        max_probability: &mut f64,
        has_learned_distribution: &mut bool,
        bit_repr: &mut BitVec,
        state: &mut StateManager,
    ) {
        while let Some(clause) = self.exploration_stack.pop() {
            if self.is_node_visitable(g, clause, comp_start, comp_size, state) {
                if !g[clause].is_learned() {
                    *hash ^= g[clause].hash();
                    *bit_repr.get_mut(g.number_variables() + clause.0).unwrap() = true;
                }
                // The clause is swap with the clause at position comp_sart + comp_size
                let current_pos = self.clause_positions[clause.0];
                let new_pos = comp_start + *comp_size;
                // Only move the nodes if it is not already in position
                // Not sure if this optimization is worth in practice
                if new_pos != current_pos {
                    let moved_node = self.clauses[new_pos];
                    self.clauses.as_mut_slice().swap(new_pos, current_pos);
                    self.clause_positions[clause.0] = new_pos;
                    self.clause_positions[moved_node.0] = current_pos;
                }
                *comp_size += 1;
                
                // Adds the variable to the hash if they have not yet been seen
                for variable in g[clause].iter_variables() {
                    if !self.seen_var[variable.0] && !g[variable].is_fixed(state) {
                        self.seen_var[variable.0] = true;
                        *hash ^= g[variable].hash();
                        *bit_repr.get_mut(variable.0).unwrap() = true;
                    }
                }

                // Explores the clauses that share a distribution with the current clause
                // TODO: Might break here
                if g[clause].has_probabilistic(state) {
                    // TODO: Might break here
                    for variable in g[clause].iter_probabilistic_variables() {
                        if !g[variable].is_fixed(state) {
                            let distribution = g[variable].distribution().unwrap();
                            *has_learned_distribution |= g[distribution].is_branching_candidate();
                            if g[distribution].is_constrained(state) && self.is_distribution_visitable(distribution, comp_distribution_start, &comp_number_distribution) {
                                let current_d_pos = self.distribution_positions[distribution.0];
                                let new_d_pos = comp_distribution_start + *comp_number_distribution;
                                if current_d_pos != new_d_pos {
                                    let moved_d = self.distributions[new_d_pos];
                                    self.distributions.as_mut_slice().swap(new_d_pos, current_d_pos);
                                    self.distribution_positions[distribution.0] = new_d_pos;
                                    self.distribution_positions[moved_d.0] = current_d_pos;
                                }
                                *max_probability *= g[distribution].remaining(state);
                                *comp_number_distribution += 1;
                                // TODO: Might also break here? 
                                for v in g[distribution].iter_variables() {
                                    if !g[v].is_fixed(state) {
                                        // TODO: Might also break here? 
                                        for c in g[v].iter_clauses_negative_occurence() {      
                                            self.exploration_stack.push(c);
                                        }
                                        // TODO: Might also break here? 
                                        for c in g[v].iter_clauses_positive_occurence() {
                                            self.exploration_stack.push(c);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Recursively explore the nodes in the connected components
                // TODO: Might also break here? 
                for parent in g[clause].iter_parents(state) {
                    self.exploration_stack.push(parent);
                }
                
                // TODO: Might also break here? 
                for child in g[clause].iter_children(state) {
                    self.exploration_stack.push(child);
                }
            }
        }
    }

    /// This function is responsible of updating the data structure with the new connected
    /// components in `g` given its current assignments.
    /// Returns true iff at least one component has been detected and it contains one distribution
    pub fn detect_components (
        &mut self,
        g: &mut Graph,
        state: &mut StateManager,
        component: ComponentIndex,
        propagator: &mut Propagator,
    ) -> bool {
        debug_assert!(propagator.unconstrained_clauses.is_empty());
        let end = state.get_usize(self.limit);
        // If we backtracked, then there are component that are not needed anymore, we truncate
        // them
        self.components.truncate(end);
        self.exploration_stack.clear();
        // Reset the set of seen variables. This can be done here and not before each component detection as
        // they do not share variables
        self.seen_var.fill(false);
        state.set_usize(self.base, end);
        let super_comp = &self.components[component.0];
        let mut start = super_comp.start;
        let mut distribution_start = super_comp.distribution_start;
        let end = start + super_comp.size;
        // We iterate over all the clause in the current component. When we encounter a constrained clause, we start
        // a component from it
        while start < end {
            let clause = self.clauses[start];
            if g[clause].is_constrained(state) {
                // If the clause is active, then we start a new component from it
                let mut size = 0;
                let mut hash: u64 = 0;
                let mut number_distribution = 0;
                let mut max_probability = 1.0;
                let mut has_learned_distribution = false;
                let mut bit_repr = bits![0].repeat(g.number_clauses_probem() + g.number_variables());
                self.exploration_stack.push(clause);
                self.explore_component(
                    g,
                    start,
                    &mut size,
                    distribution_start,
                    &mut number_distribution,
                    &mut hash,
                    &mut max_probability,
                    &mut has_learned_distribution,
                    &mut bit_repr,
                    state,
                );
                if number_distribution > 0 {
                    self.components.push(Component { start, size, distribution_start, number_distribution, hash, max_probability, has_learned_distribution, bit_repr});
                }
                distribution_start += number_distribution;
                start += size;
            } else {
                start += 1;
            }
        }
        state.set_usize(self.limit, self.components.len());
        self.number_components(state) > 0
    }

    /// Returns an iterator over the component detected by the last `detect_components` call.
    /// Note that this code is safe with the recursive nature of the solver. Since the components are
    /// added in the vector at the end, we never overwrite the components information by subsequent calls.
    /// It is only overwrite when backtracking in the search tree.
    pub fn components_iter(&self, state: &StateManager) -> ComponentIterator {
        let start = state.get_usize(self.base);
        let limit = state.get_usize(self.limit);
        ComponentIterator { limit, next: start }
    }
    
    /// Returns an iterator on the clauses of the given component
    pub fn component_iter(
        &self,
        component: ComponentIndex,
    ) -> impl Iterator<Item = ClauseIndex> + '_ {
        let start = self.components[component.0].start;
        let end = start + self.components[component.0].size;
        self.clauses[start..end].iter().copied()
    }

    pub fn find_constrained_distribution(&self, component: ComponentIndex, graph: &Graph, state: &StateManager) -> bool {
        let start = self.components[component.0].start;
        let end = start + self.components[component.0].size;
        self.clauses[start..end].iter().copied().find(|c| graph[*c].is_constrained(state) && !graph[*c].is_learned() && graph[*c].has_probabilistic(state)).is_some()
    }
    
    /// Returns the number of components
    pub fn number_components(&self, state: &StateManager) -> usize {
        state.get_usize(self.limit) - state.get_usize(self.base)
    }
    
    pub fn get_comp_hash(&self, component: ComponentIndex) -> u64 {
        self.components[component.0].hash
    }
    
    /// Returns an iterator on the distribution of a component
    pub fn component_distribution_iter(&self, component: ComponentIndex) -> impl Iterator<Item = DistributionIndex> + '_ {
        let start = self.components[component.0].distribution_start;
        let end = start + self.components[component.0].number_distribution;
        self.distributions[start..end].iter().copied()
    }

    /// Adds a clause to a component. This function is called when the solver encounters an UNSAT and needs to learn a clause.
    /// During this process we ensure that the learned clause is horn and can be safely added in the component for further detections.
    pub fn add_clause_to_component(&mut self, component: ComponentIndex, clause: ClauseIndex) {
        debug_assert!(clause.0 == self.clause_positions.len());
        let start = self.components[component.0].start;   
        self.clauses.insert(start, clause);
        for i in 0..self.clause_positions.len() {
            if self.clause_positions[i] >= start {
                self.clause_positions[i] += 1;
            }
        }
        self.clause_positions.push(start);
        for comp in self.components.iter_mut() {
            if comp.start <= start && comp.start + comp.size > start {
                comp.size += 1;
            } else if comp.start > start {
                comp.start += 1;
            }
        }
    }

    pub fn shrink(&mut self, number_clause: usize, number_variables: usize, number_distribution: usize, max_probability: f64) {
        self.clauses.truncate(number_clause);
        self.clauses.shrink_to_fit();
        self.clause_positions.truncate(number_clause);
        self.clause_positions.shrink_to_fit();
        self.seen_var.truncate(number_variables);
        self.seen_var.shrink_to_fit();
        self.distributions.truncate(number_distribution);
        self.distributions.shrink_to_fit();
        self.distribution_positions.truncate(number_distribution);
        self.distribution_positions.shrink_to_fit();
        self.components[0] = Component {
            start: 0,
            size: self.clauses.len(),
            distribution_start: 0,
            number_distribution: self.distributions.len(),
            hash: 0,
            max_probability,
            has_learned_distribution: false,
            bit_repr: bits![1].repeat(number_variables + number_clause),
        };
    }
}

impl std::ops::Index<ComponentIndex> for ComponentExtractor {
    type Output = Component;

    fn index(&self, index: ComponentIndex) -> &Self::Output {
        &self.components[index.0]
    }
}

impl std::ops::IndexMut<ComponentIndex> for ComponentExtractor {
    fn index_mut(&mut self, index: ComponentIndex) -> &mut Self::Output {
        &mut self.components[index.0]
    }
}

pub struct ComponentIterator {
    limit: usize,
    next: usize,
}

impl Iterator for ComponentIterator {
    type Item = ComponentIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.limit {
            None
        } else {
            self.next += 1;
            Some(ComponentIndex(self.next - 1))
        }
    }
}

/*
#[cfg(test)]
mod test_component_detection {
    
    use crate::core::graph::{Graph, VariableIndex, ClauseIndex};
    use crate::core::components::*;
    use search_trail::{StateManager, SaveAndRestore};
    use crate::propagator::Propagator;
    
    // Graph used for the tests:
    //
    //          C0 -> C1 ---> C2
    //                 \       |
    //                  \      v 
    //                   \--> C3 --> C4 --> C5
    fn get_graph(state: &mut StateManager) -> Graph {
        let mut g = Graph::new(state, 12, 6);
        let mut ps: Vec<VariableIndex> =(0..6).map(VariableIndex).collect();
        g.add_distributions(&vec![
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
        ], state);
        let ds = (6..12).map(VariableIndex).collect::<Vec<VariableIndex>>();
        // C0
        g.add_clause(ds[0], vec![ps[0]], state);
        // C1
        g.add_clause(ds[1], vec![ds[0], ps[1]], state);
        // C2
        g.add_clause(ds[2], vec![ds[1], ps[2]], state);
        // C3
        g.add_clause(ds[3], vec![ds[1], ds[2], ps[3]], state);
        // C4
        g.add_clause(ds[4], vec![ds[3], ps[4]], state);
        // C5
        g.add_clause(ds[5], vec![ds[4], ps[5]], state);
        g.set_variable(ds[5], false, 0, None, state);
        g
    }
    
    fn check_component(extractor: &ComponentExtractor, start: usize, end: usize, expected_clauses: Vec<ClauseIndex>) {
        let mut actual = extractor.clauses[start..end].iter().copied().collect::<Vec<ClauseIndex>>();
        actual.sort();
        assert_eq!(expected_clauses, actual);
    }

    #[test]
    fn test_initialization_extractor() {
        let mut state = StateManager::default();
        let g = get_graph(&mut state);
        let extractor = ComponentExtractor::new(&g, &mut state);
        assert_eq!(vec![0, 1, 2, 3, 4, 5], extractor.clause_positions);
        check_component(&extractor, 0, 6, (0..6).map(ClauseIndex).collect::<Vec<ClauseIndex>>());
        assert_eq!(0, state.get_usize(extractor.base));
        assert_eq!(1, state.get_usize(extractor.limit));
    }
    
    #[test]
    fn test_detect_components() {
        let mut state = StateManager::default();
        let mut g = get_graph(&mut state);
        let mut extractor = ComponentExtractor::new(&g, &mut state);
        let mut propagator = Propagator::new();

        g.set_clause_unconstrained(ClauseIndex(4), &mut state);
        extractor.detect_components(&mut g, &mut state, ComponentIndex(0), &mut propagator);

        assert_eq!(2, extractor.number_components(&state));
        assert_eq!(vec![0, 1, 2, 3, 4, 5], extractor.clause_positions);
        check_component(&extractor, 0, 4, (0..4).map(ClauseIndex).collect::<Vec<ClauseIndex>>());
        check_component(&extractor, 5, 6, vec![ClauseIndex(5)]);

        g.set_clause_unconstrained(ClauseIndex(1), &mut state);
        extractor.detect_components(&mut g, &mut state, ComponentIndex(1), &mut propagator);

        assert_eq!(2, extractor.number_components(&state));
    }

    #[test]
    fn restore_previous_state() {
        let mut state = StateManager::default();
        let mut g = get_graph(&mut state);
        let mut extractor = ComponentExtractor::new(&g, &mut state);
        let mut propagator = Propagator::new();
        
        state.save_state();

        g.set_clause_unconstrained(ClauseIndex(4), &mut state);
        extractor.detect_components(&mut g, &mut state, ComponentIndex(0), &mut propagator);

        assert_eq!(2, extractor.number_components(&state));
        assert_eq!(vec![0, 1, 2, 3, 4, 5], extractor.clause_positions);
        check_component(&extractor, 0, 4, (0..4).map(ClauseIndex).collect::<Vec<ClauseIndex>>());
        check_component(&extractor, 5, 6, vec![ClauseIndex(5)]);
        
        state.save_state();

        g.set_clause_unconstrained(ClauseIndex(1), &mut state);
        extractor.detect_components(&mut g, &mut state, ComponentIndex(1), &mut propagator);

        assert_eq!(2, extractor.number_components(&state));
        
        state.restore_state();

        assert_eq!(2, extractor.number_components(&state));
        assert_eq!(vec![0, 1, 2, 3, 4, 5], extractor.clause_positions);
        check_component(&extractor, 0, 4, (0..4).map(ClauseIndex).collect::<Vec<ClauseIndex>>());
        check_component(&extractor, 5, 6, vec![ClauseIndex(5)]);

        state.restore_state();
        
        assert_eq!(vec![0, 1, 2, 3, 4, 5], extractor.clause_positions);
        check_component(&extractor, 0, 6, (0..6).map(ClauseIndex).collect::<Vec<ClauseIndex>>());
    }
}
*/
