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

use super::graph::{ClauseIndex, Graph};
use crate::propagator::FTReachablePropagator;
use search_trail::{ReversibleUsize, StateManager, UsizeManager};

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
    positions: Vec<usize>,
    /// Holds the components computed by the extractor during the search
    components: Vec<Component>,
    /// The index of the first component of the current node in the search tree
    base: ReversibleUsize,
    /// The first index which is not a component of the current node in the search tree
    limit: ReversibleUsize,
    /// Which variable has been seen during the exploration, for the hash computation
    seen_var: Vec<bool>,
}

/// A Component is identified by two integers. The first is the index in the vector of clauses at
/// which the component starts, and the second is the size of the component.
#[derive(Debug, Clone, Copy)]
pub struct Component {
    /// First index of the component
    start: usize,
    /// Size of the component
    size: usize,
    /// Hash of the component, computed during its detection
    hash: u64,
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

impl ComponentExtractor {
    /// Creates a new component extractor for the implication graph `g`
    pub fn new(g: &Graph, state: &mut StateManager) -> Self {
        let nodes = (0..g.number_clauses()).map(ClauseIndex).collect();
        let positions = (0..g.number_clauses()).collect();
        let components = vec![Component {
            start: 0,
            size: g.number_clauses(),
            hash: 0,
        }];
        Self {
            clauses: nodes,
            positions,
            components,
            base: state.manage_usize(0),
            limit: state.manage_usize(1),
            seen_var: vec![false; g.number_variables()],
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
        let clause_pos = self.positions[clause.0];
        g.is_clause_constrained(clause, state) && !(comp_start <= clause_pos && clause_pos < (comp_start + *comp_size))
    }

    /// Adds the clause to the component and recursively explores its children, parents and clauses sharing a distribution
    /// with it. If the clause is unconstrained or has already been visited, it is skipped.
    fn explore_component(
        &mut self,
        g: &Graph,
        clause: ClauseIndex,
        comp_start: usize,
        comp_size: &mut usize,
        hash: &mut u64,
        state: &mut StateManager,
    ) {
        if self.is_node_visitable(g, clause, comp_start, comp_size, state) {
            *hash ^= g.get_clause_random(clause);
            // The clause is swap with the clause at position comp_sart + comp_size
            let current_pos = self.positions[clause.0];
            let new_pos = comp_start + *comp_size;
            // Only move the nodes if it is not already in position
            // Not sure if this optimization is worth in practice
            if new_pos != current_pos {
                let moved_node = self.clauses[new_pos];
                self.clauses.as_mut_slice().swap(new_pos, current_pos);
                self.positions[clause.0] = new_pos;
                self.positions[moved_node.0] = current_pos;
            }
            *comp_size += 1;
            
            // Adds the variable to the hash if they have not yet been seen
            for variable in g.clause_body_iter(clause, state) {
                if !self.seen_var[variable.0] && !g.is_variable_fixed(variable, state) {
                    self.seen_var[variable.0] = true;
                    *hash ^= g.get_variable_random(variable);
                }
            }
            let head = g.get_clause_head(clause);
            if !self.seen_var[head.0] && !g.is_variable_fixed(head, state) {
                self.seen_var[head.0] = true;
                *hash ^= g.get_variable_random(head);
            }

            // Explores the clauses that share a distribution with the current clause
            if g.clause_has_probabilistic(clause, state) {
                for variable in g.clause_body_iter(clause, state).chain(std::iter::once(g.get_clause_head(clause))) {
                    if g.is_variable_probabilistic(variable)
                        && !g.is_variable_fixed(variable, state)
                    {
                        let d = g.get_variable_distribution(variable).unwrap();
                        for v in g.distribution_variable_iter(d) {
                            if !g.is_variable_fixed(v, state) {
                                for c in g.variable_clause_body_iter(v) {
                                    self.explore_component(g, c, comp_start, comp_size, hash, state);
                                }
                                for c in g.variable_clause_head_iter(v) {
                                    self.explore_component(g, c, comp_start, comp_size, hash, state);
                                }
                            }
                        }
                    }
                }
            }
            
            // Recursively explore the nodes in the connected components
            for parent in g.parents_clause_iter(clause, state) {
                self.explore_component(g, parent, comp_start, comp_size, hash, state);
            }
            
            for child in g.children_clause_iter(clause, state) {
                self.explore_component(g, child, comp_start, comp_size, hash, state);
            }
        }
    }

    /// This function is responsible of updating the data structure with the new connected
    /// components in `g` given its current assignments.
    /// Returns true iff at least one component has been detected and it contains one distribution
    pub fn detect_components<const C: bool>(
        &mut self,
        g: &mut Graph,
        state: &mut StateManager,
        component: ComponentIndex,
        propagator: &mut FTReachablePropagator<C>,
    ) -> bool {
        debug_assert!(propagator.unconstrained_clauses.is_empty());
        let end = state.get_usize(self.limit);
        // If we backtracked, then there are component that are not needed anymore, we truncate
        // them
        self.components.truncate(end);
        // Reset the set of seen variables. This can be done here and not before each component detection as
        // they do not share variables
        self.seen_var.fill(false);
        state.set_usize(self.base, end);
        let super_comp = self.components[component.0];
        let mut start = super_comp.start;
        let end = start + super_comp.size;
        // We iterate over all the clause in the current component. When we encounter a constrained clause, we start
        // a component from it
        while start < end {
            let clause = self.clauses[start];
            if g.is_clause_constrained(clause, state) {
                // If the clause is active, then we start a new component from it
                let mut size = 0;
                let mut hash: u64 = 0;
                self.explore_component(
                    g,
                    clause,
                    start,
                    &mut size,
                    &mut hash,
                    state,
                );
                self.components.push(Component { start, size, hash});
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
    
    /// Returns the number of components
    pub fn number_components(&self, state: &StateManager) -> usize {
        state.get_usize(self.limit) - state.get_usize(self.base)
    }
    
    pub fn get_comp_hash(&self, component: ComponentIndex) -> u64 {
        self.components[component.0].hash
    }
}

/// This structure is used to implement a simple component detector that always returns one
/// component with all the unassigned node in it. It is used to isolate bugs annd should not be
/// used for real data sets (as it performences will be terrible)
#[allow(dead_code)]
pub struct NoComponentExtractor {
    components: Vec<Vec<ClauseIndex>>,
}

#[allow(dead_code)]
#[cfg(not(tarpaulin_include))]
impl NoComponentExtractor {
    pub fn new(g: &Graph) -> Self {
        let mut components: Vec<Vec<ClauseIndex>> = vec![];
        let nodes = g.clause_iter().collect();
    components.push(nodes);
        Self { components }
    }

    fn detect_components(
        &mut self,
        g: &Graph,
        state: &mut StateManager,
    _component: ComponentIndex,
    ) {
        let mut nodes: Vec<ClauseIndex> = vec![];
        for n in g.clause_iter() {
            if g.is_clause_constrained(n, state) {
                nodes.push(n);
            }
        }
        self.components.push(nodes);
    }

    fn get_component_hash(&self, _component: ComponentIndex) -> u64 {
        0_u64
    }

    fn components_iter(&self, _state: &StateManager) -> ComponentIterator {
        ComponentIterator {
            limit: self.components.len(),
            next: self.components.len() - 1,
        }
    }

    fn number_components(&self, _state: &StateManager) -> usize {
        1
    }

    fn component_size(&self, _component: ComponentIndex) -> usize {
        0
    }
}

#[cfg(test)]
mod test_component_detection {
    
    use crate::core::graph::{Graph, VariableIndex, ClauseIndex};
    use crate::core::components::*;
    use search_trail::{StateManager, SaveAndRestore};
    
    // Graph used for the tests:
    //
    //          C0 -> C1 ---> C2
    //                 \       |
    //                  \      v 
    //                   \--> C3 --> C4 --> C5
    fn get_graph(state: &mut StateManager) -> Graph {
        let mut g = Graph::new(state);
        let mut ps: Vec<VariableIndex> = vec![];
        for i in 0..6 {
            g.add_distribution(&vec![1.0], state);
            ps.push(VariableIndex(i))
        }
        let ds = (0..6).map(|_| g.add_variable(false, None, None, state)).collect::<Vec<VariableIndex>>();
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
        g.set_variable(ds[5], false, state);
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
        assert_eq!(vec![0, 1, 2, 3, 4, 5], extractor.positions);
        check_component(&extractor, 0, 6, (0..6).map(ClauseIndex).collect::<Vec<ClauseIndex>>());
        assert_eq!(0, state.get_usize(extractor.base));
        assert_eq!(1, state.get_usize(extractor.limit));
    }
    
    #[test]
    fn test_detect_components() {
        let mut state = StateManager::default();
        let mut g = get_graph(&mut state);
        let mut extractor = ComponentExtractor::new(&g, &mut state);
        let mut propagator = FTReachablePropagator::<false>::new();

        g.set_clause_unconstrained(ClauseIndex(4), &mut state);
        extractor.detect_components(&mut g, &mut state, ComponentIndex(0), &mut propagator);

        assert_eq!(2, extractor.number_components(&state));
        assert_eq!(vec![0, 1, 2, 3, 4, 5], extractor.positions);
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
        let mut propagator = FTReachablePropagator::<false>::new();
        
        state.save_state();

        g.set_clause_unconstrained(ClauseIndex(4), &mut state);
        extractor.detect_components(&mut g, &mut state, ComponentIndex(0), &mut propagator);

        assert_eq!(2, extractor.number_components(&state));
        assert_eq!(vec![0, 1, 2, 3, 4, 5], extractor.positions);
        check_component(&extractor, 0, 4, (0..4).map(ClauseIndex).collect::<Vec<ClauseIndex>>());
        check_component(&extractor, 5, 6, vec![ClauseIndex(5)]);
        
        state.save_state();

        g.set_clause_unconstrained(ClauseIndex(1), &mut state);
        extractor.detect_components(&mut g, &mut state, ComponentIndex(1), &mut propagator);

        assert_eq!(2, extractor.number_components(&state));
        
        state.restore_state();

        assert_eq!(2, extractor.number_components(&state));
        assert_eq!(vec![0, 1, 2, 3, 4, 5], extractor.positions);
        check_component(&extractor, 0, 4, (0..4).map(ClauseIndex).collect::<Vec<ClauseIndex>>());
        check_component(&extractor, 5, 6, vec![ClauseIndex(5)]);

        state.restore_state();
        
        assert_eq!(vec![0, 1, 2, 3, 4, 5], extractor.positions);
        check_component(&extractor, 0, 6, (0..6).map(ClauseIndex).collect::<Vec<ClauseIndex>>());
    }
}