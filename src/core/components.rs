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

//! This module provides structure used to detect connected components in a graph.
//! Two nodes u and v are connected if and only if
//!     1. There is an edge u-v in the graph (each node can be either the source or the
//!        destination)
//!     2. This edge is active
//!     or
//!     1. u and v are probabilistic nodes and in the same distribution
//!
//!
//! The main extractor is based on a simple DFS on the graph, checking if two nodes are connected
//! as defined above. During this DFS the extractor is also responsible for retrieving information
//! about the connected component being processed. These informations might be used in the
//! branching heuristics to help decompose the problem.
//! This module also provides a special extractor that do not detect any components (i.e. it always
//! returns a single component with every non-assigned node in the graph). This extractor should
//! only be used for debugging purposes.

use super::graph::{ClauseIndex, Graph};
use crate::{solver::propagator::FTReachablePropagator};
use nalgebra::DMatrix;
use search_trail::{ReversibleUsize, StateManager, UsizeManager};

/// Abstraction used as a typesafe way of retrieving a `Component`
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ComponentIndex(pub usize);

/// A Component is identified by two integers. The first is the index in the vector of nodes at
/// which the component starts, and the second is the size of the component.
#[derive(Debug, Clone, Copy)]
pub struct Component {
    /// First index of the component
    start: usize,
    /// Size of the component
    size: usize,
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

/// This structure is an extractor of component that works by doing a DFS on the component to
/// extract sub-components. It keeps all the `ClauseIndex` in a single vector (the `nodes`
/// vector) and maintain the property that all the nodes of a component are in a contiguous
/// part of the vectors. All the nodes of a component can be identified by two integers:
///     1. The index of the first node in the component
///     2. The size of the component
///
/// A second vector, `positions`, is used to keep track of the position of each node in `nodes`
/// (since they are swapped from their initial position).
/// For instance let us assume that we have two components, the `nodes` vector might looks like
/// [0, 2, 4, 1 | 5, 7 ] (the | is the separation between the two components, and integers
/// represent the nodes).
/// If now the first component is split in two, the nodes are moved in the sub-vectors spanning the
/// first four indexes. Thus (assuming 0 and 1 are in the same component and 2 and 4 in another),
/// this is a valid representation of the new vector [0, 1 | 2, 4 | 5, 7] while this is not
/// [2, 4 | 5, 7 | 0, 1].
/// During the DFS, the extractor also apply Tarjan's algorithm for finding articulation nodes
/// (nodes that, if removed, divide the graph in two disconnected part).
pub struct ComponentExtractor {
    /// The vector containing the nodes. All nodes in a component are in a contiguous part of this
    /// vector
    nodes: Vec<ClauseIndex>,
    /// The vector mapping for each `ClauseIndex` its position in `nodes`
    positions: Vec<usize>,
    /// Holds the components computed by the extractor during the search
    components: Vec<Component>,
    /// The index of the first component of the current node in the search tree
    base: ReversibleUsize,
    /// The first index which is not a component of the current node in the search tree
    limit: ReversibleUsize,
    /// Fiedler score for Fiedler-based heuristics
    fiedler_score: Vec<f64>,
}

impl ComponentExtractor {
    pub fn new(g: &Graph, state: &mut StateManager) -> Self {
        let nodes = (0..g.number_clauses()).map(ClauseIndex).collect();
        let positions = (0..g.number_clauses()).collect();
        let components = vec![Component {
            start: 0,
            size: g.number_clauses(),
        }];
        let fiedler_score: Vec<f64> = (0..g.number_clauses()).map(|_| 0.0).collect();
        Self {
            nodes,
            positions,
            components,
            base: state.manage_usize(0),
            limit: state.manage_usize(1),
            fiedler_score,
        }
    }

    /// Returns true if the node has not been visited during this DFS
    fn is_node_visitable(
        &self,
        g: &Graph,
        node: ClauseIndex,
        comp_start: usize,
        comp_size: &usize,
        state: &StateManager,
    ) -> bool {
        // If the node is bound, then it is not part of any component. In the same manner, if its
        // position is already in the part of the component that has been processed, then we must
        // not visit again
        let node_pos = self.positions[node.0];
        !(!g.is_clause_constrained(node, state)
            || (comp_start <= node_pos && node_pos < (comp_start + *comp_size)))
    }

    /// Recursively explore `node` to find all nodes in its component. If `node` has not been
    /// visited, adds its hash to the hash of the component.
    fn explore_component(
        &mut self,
        g: &Graph,
        node: ClauseIndex,
        comp_start: usize,
        comp_size: &mut usize,
        state: &mut StateManager,
        laplacians: &mut Vec<Vec<f64>>,
        laplacian_start: usize,
    ) {
        if self.is_node_visitable(g, node, comp_start, comp_size, state) {
            // The node is swap with the node at position comp_sart + comp_size
            let current_pos = self.positions[node.0];
            let new_pos = comp_start + *comp_size;
            // Only move the nodes if it is not already in position
            // Not sure if this optimization is worth in practice
            if new_pos != current_pos {
                let moved_node = self.nodes[new_pos];
                self.nodes.as_mut_slice().swap(new_pos, current_pos);
                self.positions[node.0] = new_pos;
                self.positions[moved_node.0] = current_pos;
            }
            *comp_size += 1;

            if g.clause_has_probabilistic(node, state) {
                for variable in g.clause_body_iter(node, state) {
                    if g.is_variable_probabilistic(variable)
                        && !g.is_variable_bound(variable, state)
                    {
                        let d = g.get_variable_distribution(variable).unwrap();
                        for v in g.distribution_variable_iter(d) {
                            if !g.is_variable_bound(v, state) {
                                for c in g.variable_clause_body_iter(v) {
                                    self.explore_component(
                                        g,
                                        c,
                                        comp_start,
                                        comp_size,
                                        state,
                                        laplacians,
                                        laplacian_start,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            
            // Recursively explore the nodes in the connected components
            for parent_edge in g.parents_clause_iter(node) {
                if g.is_edge_active(parent_edge, state) {
                    let parent = g.get_edge_source(parent_edge);
                    if g.is_clause_constrained(parent, state) {
                        self.explore_component(
                            g,
                            parent,
                            comp_start,
                            comp_size,
                            state,
                            laplacians,
                            laplacian_start,
                        );
                        let src_pos = self.positions[parent.0];
                        let dst_pos = self.positions[node.0];
                        laplacians[src_pos - laplacian_start][dst_pos - laplacian_start] = -1.0;
                        laplacians[dst_pos - laplacian_start][src_pos - laplacian_start] = -1.0;
                        laplacians[src_pos - laplacian_start][src_pos - laplacian_start] += 0.5;
                        laplacians[dst_pos - laplacian_start][dst_pos - laplacian_start] += 0.5;
                    }
                }
            }
            
            for child_edge in g.children_clause_iter(node) {
                if g.is_edge_active(child_edge, state) {
                    let child = g.get_edge_destination(child_edge);
                    if g.is_clause_constrained(child, state) {
                        self.explore_component(
                            g,
                            child,
                            comp_start,
                            comp_size,
                            state,
                            laplacians,
                            laplacian_start,
                        );
                        let src_pos = self.positions[node.0];
                        let dst_pos = self.positions[child.0];
                        laplacians[src_pos - laplacian_start][dst_pos - laplacian_start] = -1.0;
                        laplacians[dst_pos - laplacian_start][src_pos - laplacian_start] = -1.0;
                        laplacians[src_pos - laplacian_start][src_pos - laplacian_start] += 0.5;
                        laplacians[dst_pos - laplacian_start][dst_pos - laplacian_start] += 0.5;
                    }
                }
            }
        }
    }
    
    fn set_t_reachability(&self, g: &Graph, state: &StateManager, t_reachable: &mut Vec<bool>, pos: usize, super_comp_start: usize, super_comp_end: usize) {
        if super_comp_start <= pos && pos < super_comp_end && !t_reachable[pos - super_comp_start] {
            let clause = self.nodes[pos];
            t_reachable[pos - super_comp_start] = true;
            for child_edge in g.children_clause_iter(clause) {
                let child = g.get_edge_destination(child_edge);
                if g.is_clause_constrained(child, state) {
                    let new_pos = self.positions[child.0];
                    self.set_t_reachability(g, state, t_reachable, new_pos, super_comp_start, super_comp_end);
                }
            }
        }
    }

    fn set_f_reachability(&self, g: &Graph, state: &StateManager, f_reachable: &mut Vec<bool>, pos: usize, super_comp_start: usize, super_comp_end: usize) {
        if super_comp_start <= pos && pos < super_comp_end && !f_reachable[pos - super_comp_start] {
            f_reachable[pos - super_comp_start] = true;
            let clause = self.nodes[pos];
            for parent_edge in g.parents_clause_iter(clause) {
                let parent = g.get_edge_source(parent_edge);
                if g.is_clause_constrained(parent, state) {
                    let next_pos = self.positions[parent.0];
                    self.set_f_reachability(g, state, f_reachable, next_pos, super_comp_start, super_comp_end);
                }
            }
        }
    }
    
    fn set_reachability(&self, g: &Graph, state: &StateManager, t_reachable: &mut Vec<bool>, f_reachable: &mut Vec<bool>, super_comp_start: usize, super_comp_end: usize) {
        for i in 0..t_reachable.len() {
            let clause = self.nodes[super_comp_start + i];
            if g.clause_number_deterministic(clause, state) == 0 {
                self.set_t_reachability(g, state, t_reachable, super_comp_start + i, super_comp_start, super_comp_end);
            }
            if let Some(b) = g.get_variable_value(g.get_clause_head(clause), state) {
                if !b {
                    self.set_f_reachability(g, state, f_reachable, super_comp_start + i, super_comp_start, super_comp_end);
                }
            }
        }
    }


    /// This function is responsible of updating the data structure with the new connected
    /// components in `g` given its current assignments.
    /// Returns true iff at least one component has been detected and it contains one distribution
    pub fn detect_components(
        &mut self,
        g: &Graph,
        state: &mut StateManager,
        component: ComponentIndex,
        propagator: &mut FTReachablePropagator,
    ) -> bool {
        debug_assert!(propagator.unconstrained_clauses.is_empty());
        let c = self.components[component.0];
        for node in (c.start..(c.start + c.size)).map(ClauseIndex) {
            self.fiedler_score[node.0] = 0.0;
        }
        let end = state.get_usize(self.limit);
        // If we backtracked, then there are component that are not needed anymore, we truncate
        // them
        self.components.truncate(end);
        state.set_usize(self.base, end);
        let super_comp = self.components[component.0];
        let mut start = super_comp.start;
        let end = start + super_comp.size;
        // laplacian matrix for fiedler vector computation
        let mut laplacians: Vec<Vec<f64>> = (0..super_comp.size)
            .map(|_| (0..super_comp.size).map(|_| 0.0).collect())
            .collect();
        let mut is_t_reachable = (0..super_comp.size).map(|_| false).collect::<Vec<bool>>();
        let mut is_f_reachable = (0..super_comp.size).map(|_| false).collect::<Vec<bool>>();
        self.set_reachability(g, state,  &mut is_t_reachable, &mut is_f_reachable, super_comp.start, super_comp.start + super_comp.size);
        for i in 0..super_comp.size {
            let clause = self.nodes[super_comp.start + i];
            if !is_t_reachable[i] || !is_f_reachable[i] {
                propagator.add_unconstrained_clause(clause, g, state);
            }
        }
        // We iterate over all the nodes in the component
        while start < end {
            let node = self.nodes[start];
            if g.is_clause_constrained(node, state) {
                // If the clause is active, then we start a new component from it
                let mut size = 0;
                self.explore_component(
                    g,
                    node,
                    start,
                    &mut size,
                    state,
                    &mut laplacians,
                    super_comp.start,
                );
                self.components.push(Component { start, size });
                start += size;
            } else {
                start += 1;
            }
        }
        state.set_usize(self.limit, self.components.len());
        // If there is only one sub-component extracted, we keep the same values for the fiedler
        // heuristic. Since the computation of the fiedler vector is expensive, we try to delay it
        // as much as possible. If the component was not breaked, we assume that the nodes that were
        // on the "inside" of the graph should have lower value in order to break the graph.
        if self.number_components(state) > 1 || component == ComponentIndex(0) {
            for cid in self.components_iter(state) {
                let comp = self.components[cid.0];
                let size = comp.size;
                // Extracts the laplacian matrix corresponding to this component
                let sub_lp = DMatrix::from_fn(size, size, |r, c| {
                    laplacians[(comp.start - super_comp.start) + r]
                        [(comp.start - super_comp.start) + c]
                });
                let decomp = sub_lp.hermitian_part().symmetric_eigen();
                // Finds the fiedler vectors. This is the eigenvector associated with the second
                // smallest eigenvalue. We first find this eigen value and then assign the fiedler
                // score to the nodes in the component
                let mut smallest_eigenvalue = f64::INFINITY;
                let mut second_smallest_eigenvalue = f64::INFINITY;
                let mut smallest_index = 0;
                let mut fiedler_index = 0;
                let eigenvalues = decomp.eigenvalues;
                for i in 0..size {
                    let eigenvalue = eigenvalues[i];
                    if eigenvalue < smallest_eigenvalue {
                        second_smallest_eigenvalue = smallest_eigenvalue;
                        fiedler_index = smallest_index;
                        smallest_eigenvalue = eigenvalue;
                        smallest_index = i;
                    } else if eigenvalue < second_smallest_eigenvalue {
                        second_smallest_eigenvalue = eigenvalue;
                        fiedler_index = i;
                    }
                }
                for i in 0..size {
                    self.fiedler_score[self.nodes[comp.start + i].0] =
                        decomp.eigenvectors.row(i)[fiedler_index];
                }
            }
        }
        self.number_components(state) > 0
    }

    /// Returns an iterator over the component detected by the last `detect_components` call
    pub fn components_iter(&self, state: &StateManager) -> ComponentIterator {
        let start = state.get_usize(self.base);
        let limit = state.get_usize(self.limit);
        ComponentIterator { limit, next: start }
    }
    
    /// Returns an iterator on the nodes of the given component
    pub fn component_iter(
        &self,
        component: ComponentIndex,
    ) -> impl Iterator<Item = ClauseIndex> + '_ {
        let start = self.components[component.0].start;
        let end = start + self.components[component.0].size;
        self.nodes[start..end].iter().copied()
    }
    
    /// Returns the fiedler value of a given clause
    pub fn fiedler_value(&self, node: ClauseIndex) -> f64 {
        self.fiedler_score[node.0]
    }

    /// Returns the number of components
    pub fn number_components(&self, state: &StateManager) -> usize {
        state.get_usize(self.limit) - state.get_usize(self.base)
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

/*
#[cfg(test)]
mod test_dfs_component {
    use super::{ComponentExtractor, ComponentIndex};
    use crate::core::graph::{DistributionIndex, Graph, NodeIndex};
    use crate::core::trail::{IntManager, SaveAndRestore, StateManager};
    use rustc_hash::FxHashSet;

    #[test]
    fn test_initialization_extractor() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        g.add_distribution(&vec![0.0], &mut state);
        let np = vec![NodeIndex(0)];
        let n = (0..4)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3], np[0]], &mut state);

        let mut component_extractor = ComponentExtractor::new(&g, &mut state);
        component_extractor.detect_components(&g, &mut state, ComponentIndex(0));
        assert_eq!(5, component_extractor.nodes.len());
        assert_eq!(5, component_extractor.positions.len());
        assert_eq!(1, component_extractor.distributions[0].len());
        assert_eq!(1, state.get_int(component_extractor.base));
        assert_eq!(2, state.get_int(component_extractor.limit));
        let number_component =
            state.get_int(component_extractor.limit) - state.get_int(component_extractor.base);
        assert_eq!(1, number_component);
    }

    #[test]
    fn test_initialization_extractor_distribution() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        let d = g.add_distribution(&vec![0.3, 0.4, 0.3], &mut state);
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3], n[4]], &mut state);
        g.add_clause(n[1], &vec![d[0]], &mut state);
        g.add_clause(n[3], &vec![d[1], d[2]], &mut state);
        let mut component_extractor = ComponentExtractor::new(&g, &mut state);
        component_extractor.detect_components(&g, &mut state, ComponentIndex(0));
        assert_eq!(8, component_extractor.nodes.len());
        assert_eq!(8, component_extractor.positions.len());
        assert_eq!(1, component_extractor.distributions[0].len());
        assert_eq!(1, state.get_int(component_extractor.base));
        assert_eq!(2, state.get_int(component_extractor.limit));
        let number_component =
            state.get_int(component_extractor.limit) - state.get_int(component_extractor.base);
        assert_eq!(1, number_component);
    }

    #[test]
    fn test_initialization_single_component() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        g.add_distribution(&vec![0.5], &mut state);
        let n = (0..4)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3], NodeIndex(0)], &mut state);

        let mut component_extractor = ComponentExtractor::new(&g, &mut state);
        component_extractor.detect_components(&g, &mut state, ComponentIndex(0));
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(1, components.len());
        let number_component =
            state.get_int(component_extractor.limit) - state.get_int(component_extractor.base);
        assert_eq!(1, number_component);

        let c0 = component_extractor.components[components[0].0];
        let nodes = component_extractor.nodes[c0.start..(c0.start + c0.size)]
            .iter()
            .copied()
            .collect::<FxHashSet<NodeIndex>>();
        let mut expected = n.iter().copied().collect::<FxHashSet<NodeIndex>>();
        expected.insert(NodeIndex(0));
        assert_eq!(expected, nodes);
    }

    #[test]
    fn test_initialization_multiple_components() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        g.add_distribution(&vec![0.5], &mut state);
        g.add_distribution(&vec![0.5], &mut state);
        let np = vec![NodeIndex(0), NodeIndex(1)];
        let n = (0..3)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![np[0]], &mut state);
        g.add_clause(n[1], &vec![np[1], n[2]], &mut state);

        let mut component_extractor = ComponentExtractor::new(&g, &mut state);
        component_extractor.detect_components(&g, &mut state, ComponentIndex(0));
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(2, components.len());
        let number_component =
            state.get_int(component_extractor.limit) - state.get_int(component_extractor.base);
        assert_eq!(2, number_component);
        let c0 = component_extractor.components[components[0].0];
        let n_c1 = component_extractor.nodes[c0.start..(c0.start + c0.size)]
            .iter()
            .copied()
            .collect::<FxHashSet<NodeIndex>>();
        let n1 = vec![n[0], np[0]]
            .iter()
            .copied()
            .collect::<FxHashSet<NodeIndex>>();
        let n2 = vec![n[1], n[2], np[1]]
            .iter()
            .copied()
            .collect::<FxHashSet<NodeIndex>>();
        let c1 = component_extractor.components[components[1].0];
        let n_c2 = component_extractor.nodes[c1.start..(c1.start + c1.size)]
            .iter()
            .copied()
            .collect::<FxHashSet<NodeIndex>>();
        assert_eq!(2, n_c1.len());
        assert_eq!(n1, n_c1);
        assert_eq!(3, n_c2.len());
        assert_eq!(n2, n_c2);
    }f

    #[test]
    fn test_breaking_components() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        // np[0] --
        //         | - n[1]  --
        // n[0]  --            | - n[2]
        //             np[1] --
        //
        g.add_distribution(&vec![0.5], &mut state);
        g.add_distribution(&vec![0.5], &mut state);
        let n = (0..3)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        let np = vec![NodeIndex(0), NodeIndex(1)];
        g.add_clause(n[1], &vec![np[0], n[0]], &mut state);
        g.add_clause(n[2], &vec![n[1], np[1]], &mut state);

        let mut component_extractor = ComponentExtractor::new(&g, &mut state);
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(1, components.len());
        let number_component =
            state.get_int(component_extractor.limit) - state.get_int(component_extractor.base);
        assert_eq!(1, number_component);

        g.set_node(n[1], true, &mut state);
        component_extractor.detect_components(&g, &mut state, components[0]);
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        let number_component =
            state.get_int(component_extractor.limit) - state.get_int(component_extractor.base);
        assert_eq!(2, number_component);
        assert_eq!(2, components.len());
    }

    #[test]
    fn test_breaking_component_but_backtrack() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        g.add_distribution(&vec![0.5], &mut state);
        g.add_distribution(&vec![0.5], &mut state);
        let np = vec![NodeIndex(0), NodeIndex(1)];
        let n = (0..3)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[1], &vec![np[0], n[0]], &mut state);
        g.add_clause(n[2], &vec![np[1], n[1]], &mut state);

        let mut component_extractor = ComponentExtractor::new(&g, &mut state);
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(1, components.len());
        let number_component =
            state.get_int(component_extractor.limit) - state.get_int(component_extractor.base);
        assert_eq!(1, number_component);

        state.save_state();

        g.set_node(n[1], true, &mut state);
        component_extractor.detect_components(&g, &mut state, components[0]);
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(2, components.len());
        let number_component =
            state.get_int(component_extractor.limit) - state.get_int(component_extractor.base);
        assert_eq!(2, number_component);

        state.restore_state();
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(1, components.len());
        let number_component =
            state.get_int(component_extractor.limit) - state.get_int(component_extractor.base);
        assert_eq!(1, number_component);
    }

    #[test]
    fn distributions_in_components() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        // d1[0] --
        // d1[1] -- | - n[0]
        // d1[2] --       |
        //              n[1]
        // d3[0] --------|
        // d3[1] --------|
        //              d2[0]
        let n = (0..2)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        let d1 = g.add_distribution(&vec![0.3, 0.3, 0.4], &mut state);
        let d2 = g.add_distribution(&vec![0.5, 0.5], &mut state);
        let d3 = g.add_distribution(&vec![0.3, 0.7], &mut state);

        g.add_clause(n[0], &d1, &mut state);
        g.add_clause(n[0], &vec![n[1]], &mut state);
        g.add_clause(n[1], &d3, &mut state);
        g.add_clause(n[1], &vec![d2[0]], &mut state);

        let mut component_extractor = ComponentExtractor::new(&g, &mut state);
        component_extractor.detect_components(&g, &mut state, ComponentIndex(0));
        let components: Vec<ComponentIndex> = component_extractor.components_iter(&state).collect();
        assert_eq!(1, components.len());
        let distributions = component_extractor.get_component_distributions(components[0]);
        assert_eq!(3, distributions.len());
        assert!(distributions.contains(&DistributionIndex(0)));
        assert!(distributions.contains(&DistributionIndex(1)));
        assert!(distributions.contains(&DistributionIndex(2)));

        g.set_node(n[0], false, &mut state);

        component_extractor.detect_components(&g, &mut state, components[0]);
        let components: Vec<ComponentIndex> = component_extractor.components_iter(&state).collect();
        assert_eq!(1, components.len());
        let distribution_comp2 = component_extractor.get_component_distributions(components[0]);
        assert_eq!(2, distribution_comp2.len());
        assert!(distribution_comp2.contains(&DistributionIndex(1)));
        assert!(distribution_comp2.contains(&DistributionIndex(2)));
    }
}

#[cfg(test)]
mod test_fiedler {
    use super::{ComponentExtractor, ComponentIndex};
    use crate::core::graph::{Graph, NodeIndex};
    use crate::core::trail::StateManager;
    use assert_float_eq::*;
    use nalgebra::Matrix5;

    #[test]
    fn test_one_component() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        g.add_distribution(&vec![0.5], &mut state);
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[3], &vec![NodeIndex(0)], &mut state);
        g.add_clause(n[0], &vec![NodeIndex(0)], &mut state);
        g.add_clause(n[0], &vec![n[3]], &mut state);
        g.add_clause(n[2], &vec![n[3]], &mut state);
        g.add_clause(n[1], &vec![n[0]], &mut state);
        g.add_clause(n[2], &vec![n[1]], &mut state);
        g.add_clause(n[4], &vec![n[2]], &mut state);
        let laplacian = Matrix5::from_vec(vec![
            3.0, -1.0, 0.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, -1.0, 3.0, -1.0, -1.0, -1.0,
            0.0, -1.0, 3.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0,
        ]);
        let decomp = laplacian.hermitian_part().symmetric_eigen();
        let mut smallest_eigenvalue = f64::INFINITY;
        let mut second_smallest_eigenvalue = f64::INFINITY;
        let mut smallest_index = 0;
        let mut fiedler_index = 0;
        let eigenvalues = decomp.eigenvalues;
        for i in 0..5 {
            let eigenvalue = eigenvalues[i];
            if eigenvalue < smallest_eigenvalue {
                second_smallest_eigenvalue = smallest_eigenvalue;
                fiedler_index = smallest_index;
                smallest_eigenvalue = eigenvalue;
                smallest_index = i;
            } else if eigenvalue < second_smallest_eigenvalue {
                second_smallest_eigenvalue = eigenvalue;
                fiedler_index = i;
            }
        }

        let mut component_extractor = ComponentExtractor::new(&g, &mut state);
        component_extractor.detect_components(&g, &mut state, ComponentIndex(0));
        assert_float_relative_eq!(
            0.0,
            component_extractor.fiedler_score[component_extractor.positions[0]],
            0.01
        );
        assert_float_relative_eq!(
            decomp.eigenvectors.row(0)[fiedler_index],
            component_extractor.fiedler_score[component_extractor.positions[1]],
            0.01
        );
        assert_float_relative_eq!(
            decomp.eigenvectors.row(1)[fiedler_index],
            component_extractor.fiedler_score[component_extractor.positions[2]],
            0.01
        );
        assert_float_relative_eq!(
            decomp.eigenvectors.row(2)[fiedler_index],
            component_extractor.fiedler_score[component_extractor.positions[3]],
            0.01
        );
        assert_float_relative_eq!(
            decomp.eigenvectors.row(3)[fiedler_index],
            component_extractor.fiedler_score[component_extractor.positions[4]],
            0.01
        );
        assert_float_relative_eq!(
            decomp.eigenvectors.row(4)[fiedler_index],
            component_extractor.fiedler_score[component_extractor.positions[5]],
            0.01
        );
    }
}

*/