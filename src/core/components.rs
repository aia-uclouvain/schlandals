//! This module provides structure used to detect connected components in the implication problem.
//! Two clauses C1 = (I1, h1) and C2 = (I2, h2) are connected if and only if
//!     1. Both clauses are still constrained in the implication problem
//!     2. Either h2 is in I1 or h1 is in I2
//!     3. C1 and C2 have probabilistic variables in their bodies that are from the same distribution
//!   
//! The components are extracted using a simple DSF on the implication problem. During this traversal, the
//! hash of the components are also computing.
//! This hash is a simple XOR between random bitstring that are assigned durnig the problem creation.
//! For convenience the DFS extractor also sets the f-reachability and t-reachability once the component is extracted.
//! 
//! This module also exposes a special component extractor that do not detect any components.
//! It should only be used for debugging purpose to isolate bugs

use super::problem::{ClauseIndex, VariableIndex, Problem, DistributionIndex};
use search_trail::{ReversibleUsize, StateManager, UsizeManager};
use malachite::rational::Rational;
use crate::common::rational;
use crate::caching::*;

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
    caching_scheme: CachingScheme,
    clauses_cache: Vec<ClauseIndex>,
    variables_cache: Vec<VariableIndex>,
}

/// A Component is identified by two integers. The first is the index in the vector of clauses at
/// which the component starts, and the second is the size of the component.
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
    max_probability: Rational,
    /// Representation of the component for hash
    key: CacheKey,
}

impl Component {

    pub fn max_probability(&self) -> Rational {
        self.max_probability.clone()
    }

    pub fn get_cache_key(&self) -> CacheKey {
        self.key.clone()
    }

    pub fn number_distribution(&self) -> usize {
        self.number_distribution
    }
}

impl ComponentExtractor {
    /// Creates a new component extractor for the implication problem `g`
    pub fn new(g: &Problem, caching_scheme: CachingScheme, state: &mut StateManager) -> Self {
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
            max_probability: rational(1.0),
            key: CacheKey::default(),
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
            caching_scheme,
            clauses_cache: vec![],
            variables_cache: vec![],
        }
    }

    /// Returns true if and only if:
    ///     1. The clause is constrained
    ///     2. It is part of the current component being splitted in sub-components
    ///     3. It has not yet been added to the component
    fn is_node_visitable(
        &self,
        g: &Problem,
        clause: ClauseIndex,
        comp_start: usize,
        comp_size: &usize,
        state: &StateManager,
    ) -> bool {
        if g[clause].is_learned() {
            return false;
        }
        // if the clause has already been visited, then its position in the component must
        // be between [start..(start + size)].
        let clause_pos = self.clause_positions[clause.0];
        g[clause].is_active(state) && !(comp_start <= clause_pos && clause_pos < (comp_start + *comp_size))
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
        g: &Problem,
        comp_start: usize,
        comp_size: &mut usize,
        comp_distribution_start: usize,
        comp_number_distribution: &mut usize,
        hash: &mut u64,
        max_probability: &mut Rational,
        state: &mut StateManager,
    ) {
        while let Some(clause) = self.exploration_stack.pop() {
            if self.is_node_visitable(g, clause, comp_start, comp_size, state) {
                *hash ^= g[clause].hash();
                if g[clause].is_modified(state) {
                    self.clauses_cache.push(clause);
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
                    if !g[variable].is_fixed(state) && !self.seen_var[variable.0] {
                        self.seen_var[variable.0] = true;
                        *hash ^= g[variable].hash();
                        self.variables_cache.push(variable);
                        if g[variable].is_probabilitic() {
                            let distribution = g[variable].distribution().unwrap();
                            if g[distribution].is_constrained(state) && self.is_distribution_visitable(distribution, comp_distribution_start, comp_number_distribution) {
                                let current_d_pos = self.distribution_positions[distribution.0];
                                let new_d_pos = comp_distribution_start + *comp_number_distribution;
                                if current_d_pos != new_d_pos {
                                    let moved_d = self.distributions[new_d_pos];
                                    self.distributions.as_mut_slice().swap(new_d_pos, current_d_pos);
                                    self.distribution_positions[distribution.0] = new_d_pos;
                                    self.distribution_positions[moved_d.0] = current_d_pos;
                                }
                                *max_probability *= rational(g[distribution].remaining(state));
                                *comp_number_distribution += 1;
                                for v in g[distribution].iter_variables() {
                                    if !g[v].is_fixed(state) {
                                        for c in g[v].iter_clauses_negative_occurence(state).filter(|c| !g[*c].is_learned()) {
                                            self.exploration_stack.push(c);
                                        }
                                        for c in g[v].iter_clauses_positive_occurence(state).filter(|c| !g[*c].is_learned()) {
                                            self.exploration_stack.push(c);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for c in g[clause].iter_parents(state) {
                    self.exploration_stack.push(c);
                }
                for c in g[clause].iter_children(state) {
                    self.exploration_stack.push(c);
                }
            }
        }
    }

    /// This function is responsible of updating the data structure with the new connected
    /// components in `g` given its current assignments.
    /// Returns true iff at least one component has been detected and it contains one distribution
    pub fn detect_components (
        &mut self,
        g: &mut Problem,
        state: &mut StateManager,
        component: ComponentIndex,
    ) -> bool {
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
            if g[clause].is_active(state) {
                // If the clause is active, then we start a new component from it
                let mut size = 0;
                let mut hash: u64 = 0;
                let mut number_distribution = 0;
                let mut max_probability = rational(1.0);
                self.exploration_stack.push(clause);
                self.clauses_cache.clear();
                self.variables_cache.clear();
                self.explore_component(
                    g,
                    start,
                    &mut size,
                    distribution_start,
                    &mut number_distribution,
                    &mut hash,
                    &mut max_probability,
                    state,
                );
                if number_distribution > 0 {
                    self.clauses_cache.sort();
                    self.variables_cache.sort();
                    let key = self.caching_scheme.get_key(g, &self.clauses_cache, &self.variables_cache, hash, state);
                    self.components.push(Component { start, size, distribution_start, number_distribution, hash, max_probability, key});
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

    pub fn shrink(&mut self, number_clause: usize, number_variables: usize, number_distribution: usize, max_probability: Rational) {
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
            key: CacheKey::default(),
        };
        self.caching_scheme.init(number_clause, number_variables);
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
    
    use crate::core::problem::{Problem, VariableIndex, ClauseIndex};
    use crate::core::components::*;
    use search_trail::{StateManager, SaveAndRestore};
    use crate::propagator::Propagator;
    
    // Problem used for the tests:
    //
    //          C0 -> C1 ---> C2
    //                 \       |
    //                  \      v 
    //                   \--> C3 --> C4 --> C5
    fn get_problem(state: &mut StateManager) -> Problem {
        let mut g = Problem::new(state, 12, 6);
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
        let g = get_problem(&mut state);
        let extractor = ComponentExtractor::new(&g, &mut state);
        assert_eq!(vec![0, 1, 2, 3, 4, 5], extractor.clause_positions);
        check_component(&extractor, 0, 6, (0..6).map(ClauseIndex).collect::<Vec<ClauseIndex>>());
        assert_eq!(0, state.get_usize(extractor.base));
        assert_eq!(1, state.get_usize(extractor.limit));
    }
    
    #[test]
    fn test_detect_components() {
        let mut state = StateManager::default();
        let mut g = get_problem(&mut state);
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
        let mut g = get_problem(&mut state);
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
