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

use super::graph::{DistributionIndex, Graph, NodeIndex};
use super::trail::*;
use nalgebra::DMatrix;
use rustc_hash::{FxHashSet, FxHasher};
use std::hash::Hasher;

/// Abstraction used as a typesafe way of retrieving a `Component`
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ComponentIndex(pub usize);

/// A Component is identified by two integers. The first is the index in the vector of nodes at
/// which the component starts, and the second is the size of the component.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Component {
    /// First index of the component
    start: usize,
    /// Size of the component
    size: usize,
    /// Hash of the component (computed during the detection, or afterward)
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

/// This structure is an extractor of component that works by doing a DFS on the component to
/// extract sub-components. It keeps all the `NodeIndex` in a single vector (the `nodes`
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
    nodes: Vec<NodeIndex>,
    /// The vector mapping for each `NodeIndex` its position in `nodes`
    positions: Vec<usize>,
    /// Holds the components computed by the extractor during the search
    components: Vec<Component>,
    /// Holds the distribution in each component of `components`
    distributions: Vec<FxHashSet<DistributionIndex>>,
    /// The index of the first component of the current node in the search tree
    base: ReversibleInt,
    /// The first index which is not a component of the current node in the search tree
    limit: ReversibleInt,
    // Vector of random 64-bits number for the hash computation when a node is in a component
    nodes_bits: Vec<u64>,
    // Vector of random 64-bits number for the hash computation when an edge is in a component
    edges_bits: Vec<u64>,
    /// Parents for the bicomponent detection
    parents: Vec<Option<NodeIndex>>,
    /// Depth for the bicomponent detection
    depth: Vec<usize>,
    /// Lowest for the bicomponent detection
    low: Vec<usize>,
    /// Count for each distribution the number of articulation point in it
    ap_heuristic_score: Vec<usize>,
    /// Fiedler score for Fiedler-based heuristics
    fiedler_score: Vec<f64>,
}

impl ComponentExtractor {
    pub fn new(g: &Graph, state: &mut StateManager) -> Self {
        let nodes = (0..g.number_nodes()).map(NodeIndex).collect();
        let positions = (0..g.number_nodes()).collect();
        let components = vec![Component {
            start: 0,
            size: g.number_nodes(),
            hash: 0,
        }];
        let first_distributions = (0..g.number_distributions())
            .map(DistributionIndex)
            .collect::<FxHashSet<DistributionIndex>>();
        let nodes_bits: Vec<u64> = (0..g.number_nodes()).map(|_| rand::random()).collect();
        let edges_bits: Vec<u64> = (0..g.number_edges()).map(|_| rand::random()).collect();
        let parents: Vec<Option<NodeIndex>> = (0..g.number_nodes()).map(|_| None).collect();
        let depth: Vec<usize> = (0..g.number_nodes()).map(|_| 0).collect();
        let low: Vec<usize> = (0..g.number_nodes()).map(|_| 0).collect();
        let ap_heuristic_score: Vec<usize> = (0..g.number_distributions()).map(|_| 0).collect();
        let fiedler_score: Vec<f64> = (0..g.number_nodes()).map(|_| 0.0).collect();
        Self {
            nodes,
            positions,
            components,
            distributions: vec![first_distributions],
            base: state.manage_int(0),
            limit: state.manage_int(1),
            nodes_bits,
            edges_bits,
            parents,
            depth,
            low,
            ap_heuristic_score,
            fiedler_score,
        }
    }

    /// Returns true if the node has not been visited during this DFS
    fn is_node_visitable(
        &self,
        g: &Graph,
        node: NodeIndex,
        comp_start: usize,
        comp_size: &usize,
        state: &StateManager,
    ) -> bool {
        // If the node is bound, then it is not part of any component. In the same manner, if its
        // position is already in the part of the component that has been processed, then we must
        // not visit again
        let node_pos = self.positions[node.0];
        !(g.is_node_bound(node, state)
            || (comp_start <= node_pos && node_pos < (comp_start + *comp_size)))
    }

    /// Recursively explore `node` to find all nodes in its component. If `node` has not been
    /// visited, adds its hash to the hash of the component.
    fn explore_component(
        &mut self,
        g: &Graph,
        node: NodeIndex,
        comp_start: usize,
        comp_size: &mut usize,
        state: &mut StateManager,
        hash: &mut u64,
        depth: usize,
        laplacians: &mut Vec<Vec<f64>>,
        laplacian_start: usize,
    ) {
        if self.is_node_visitable(g, node, comp_start, comp_size, state) {
            // Adds the node random bits to the hash
            *hash ^= self.nodes_bits[node.0];
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

            // Update the vectors for biconnected component detection
            self.depth[node.0] = depth;
            self.low[node.0] = depth;
            let mut child_count = 0;
            let mut is_articulation = false;

            // If the node is probabilistic, add its distribution to the set of distribution for this
            // component
            // And we need to add the nodes in the distribution in the component
            if g.is_node_probabilistic(node) {
                let distribution = g.get_distribution(node).unwrap();
                self.distributions.last_mut().unwrap().insert(distribution);
                for n in g.distribution_iter(distribution) {
                    if self.is_node_visitable(g, n, comp_start, comp_size, state) {
                        self.parents[n.0] = Some(node);
                        self.explore_component(
                            g,
                            n,
                            comp_start,
                            comp_size,
                            state,
                            hash,
                            depth + 1,
                            laplacians,
                            laplacian_start,
                        );
                        child_count += 1;
                        if self.low[n.0] >= self.depth[node.0] {
                            is_articulation = true;
                        }
                    } else if self.parents[node.0].is_some() && n != self.parents[node.0].unwrap() {
                        self.low[node.0] = self.low[node.0].min(self.depth[n.0]);
                    }
                }
            }

            // Recursively explore the nodes in the connected components (i.e. linked by a clause)
            for clause in g.node_clauses(node) {
                for edge in g.edges_clause(clause) {
                    if g.is_edge_active(edge, state) {
                        if node == g.get_edge_source(edge) {
                            *hash ^= self.edges_bits[edge.0];
                        }
                        let src = g.get_edge_source(edge);
                        let dst = g.get_edge_destination(edge);
                        if self.is_node_visitable(g, src, comp_start, comp_size, state) {
                            self.parents[src.0] = Some(node);
                            self.explore_component(
                                g,
                                src,
                                comp_start,
                                comp_size,
                                state,
                                hash,
                                depth + 1,
                                laplacians,
                                laplacian_start,
                            );
                            child_count += 1;
                            if self.low[src.0] >= self.depth[node.0] {
                                is_articulation = true;
                            }
                        } else if let Some(parent) = self.parents[node.0] {
                            if parent != node {
                                self.low[node.0] = self.low[node.0].min(self.depth[src.0]);
                            }
                        }
                        if self.is_node_visitable(g, dst, comp_start, comp_size, state) {
                            self.parents[dst.0] = Some(node);
                            self.explore_component(
                                g,
                                dst,
                                comp_start,
                                comp_size,
                                state,
                                hash,
                                depth + 1,
                                laplacians,
                                laplacian_start,
                            );
                            child_count += 1;
                            if self.low[dst.0] >= self.depth[node.0] {
                                is_articulation = true;
                            }
                        } else if let Some(parent) = self.parents[node.0] {
                            if parent != node {
                                self.low[node.0] = self.low[node.0].min(self.depth[dst.0]);
                            }
                        }
                        if !g.is_node_bound(src, state) && !g.is_node_bound(dst, state) {
                            let src_pos = self.positions[src.0];
                            let dst_pos = self.positions[dst.0];
                            if g.is_node_deterministic(src) && g.is_node_deterministic(dst) {
                                laplacians[src_pos - laplacian_start][dst_pos - laplacian_start] =
                                    -1.0;
                                laplacians[dst_pos - laplacian_start][src_pos - laplacian_start] =
                                    -1.0;
                                laplacians[src_pos - laplacian_start][src_pos - laplacian_start] +=
                                    0.5;
                                laplacians[dst_pos - laplacian_start][dst_pos - laplacian_start] +=
                                    0.5;
                            }
                        }
                    }
                }
            }
            if (self.parents[node.0].is_some() && is_articulation)
                || (self.parents[node.0].is_none() && child_count > 1)
            {
                if g.is_node_probabilistic(node) {
                    self.ap_heuristic_score[g.get_distribution(node).unwrap().0] += 1;
                } else {
                    for p in g.incomings(node).map(|edge| g.get_edge_source(edge)) {
                        if g.is_node_probabilistic(p) {
                            self.ap_heuristic_score[g.get_distribution(p).unwrap().0] += 1;
                        }
                    }
                }
            }
        }
    }

    /// This function is responsible of updating the data structure with the new connected
    /// components in `g` given its current assignments.
    pub fn detect_components(
        &mut self,
        g: &Graph,
        state: &mut StateManager,
        component: ComponentIndex,
    ) {
        let c = self.components[component.0];
        for node in (c.start..(c.start + c.size)).map(NodeIndex) {
            self.parents[node.0] = None;
            // Reset the articulation point heuristic for this super-component before detecting its
            // new sub-components
            if g.is_node_probabilistic(node) {
                self.ap_heuristic_score[g.get_distribution(node).unwrap().0] = 0;
            }
            self.fiedler_score[node.0] = 0.0;
        }
        let end = state.get_int(self.limit);
        // If we backtracked, then there are component that are not needed anymore, we truncate
        // them
        self.components.truncate(end as usize);
        self.distributions.truncate(end as usize);
        state.set_int(self.base, end);
        let comp = self.components[component.0];
        let mut start = comp.start;
        let end = start + comp.size;
        // laplacian matrix for fiedler vector computation
        let mut laplacians: Vec<Vec<f64>> = (0..comp.size)
            .map(|_| (0..comp.size).map(|_| 0.0).collect())
            .collect();
        // We iterate over all the nodes in the component
        while start < end {
            let node = self.nodes[start];
            if !g.is_node_bound(node, state) {
                // If the node is not bound, then we start a new component from it
                let mut size = 0;
                self.distributions.push(FxHashSet::default());
                let mut hash: u64 = 0;
                self.explore_component(
                    g,
                    node,
                    start,
                    &mut size,
                    state,
                    &mut hash,
                    0,
                    &mut laplacians,
                    comp.start,
                );
                self.components.push(Component { start, size, hash });
                let sub_lp = DMatrix::from_fn(size, size, |r, c| {
                    laplacians[(start - comp.start) + r][(start - comp.start) + c]
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
                    let node = self.nodes[i + comp.start];
                    let pos = self.positions[node.0];
                    self.fiedler_score[pos] =
                        decomp.eigenvectors.row(pos - comp.start)[fiedler_index];
                }
                start += size;
            } else {
                start += 1;
            }
        }
        state.set_int(self.limit, self.components.len() as isize);
    }

    /// Returns the hash of a given component. This is the job of the implementing data structure
    /// to ensure that the same component gives the same hash, even if it appears in another part
    /// of the search tree. This means that, in practice, the hash should give the same hash even
    /// if the "order" of the nodes and/or edges in the component is changed.
    pub fn get_component_hash(&self, component: ComponentIndex) -> u64 {
        let mut hasher = FxHasher::default();
        hasher.write_u64(self.components[component.0].hash);
        hasher.finish()
    }

    /// Returns the distributions in a given component
    pub fn get_component_distributions(
        &self,
        component: ComponentIndex,
    ) -> &FxHashSet<DistributionIndex> {
        &self.distributions[component.0]
    }

    /// Returns an iterator over the component detected by the last `detect_components` call
    pub fn components_iter(&self, state: &StateManager) -> ComponentIterator {
        let start = state.get_int(self.base) as usize;
        let limit = state.get_int(self.limit) as usize;
        ComponentIterator { limit, next: start }
    }

    /// Returns the number of articulation point in `distribution`
    pub fn get_distribution_ap_score(&self, distribution: DistributionIndex) -> usize {
        self.ap_heuristic_score[distribution.0]
    }

    pub fn get_distribution_fiedler_neighbor_diff_avg(
        &self,
        g: &Graph,
        distribution: DistributionIndex,
        state: &StateManager,
    ) -> f64 {
        let mut score = 0.0;
        let mut nb_distribution = 0.0;
        for node in g
            .distribution_iter(distribution)
            .filter(|n| !g.is_node_bound(*n, state))
        {
            nb_distribution += 1.0;
            let node_fiedler_value = self.fiedler_score[self.positions[node.0]];
            let mut diff = 0.0;
            let mut nb_active_children = 0.0;
            for children in g.active_children(node, state) {
                let child_fiedler_value = self.fiedler_score[self.positions[children.0]];
                diff += (node_fiedler_value - child_fiedler_value).abs();
                nb_active_children += 1.0;
            }
            if nb_active_children != 0.0 {
                score += diff / nb_active_children
            }
        }
        score / nb_distribution
    }

    /// Returns the number of components
    pub fn number_components(&self, state: &StateManager) -> usize {
        (state.get_int(self.limit) - state.get_int(self.base)) as usize
    }
}

/// This structure is used to implement a simple component detector that always returns one
/// component with all the unassigned node in it. It is used to isolate bugs annd should not be
/// used for real data sets (as it performences will be terrible)
#[allow(dead_code)]
pub struct NoComponentExtractor {
    components: Vec<Vec<NodeIndex>>,
    distributions: Vec<FxHashSet<DistributionIndex>>,
}

#[allow(dead_code)]
impl NoComponentExtractor {
    pub fn new(g: &Graph) -> Self {
        let mut components: Vec<Vec<NodeIndex>> = vec![];
        let nodes = g.nodes_iter().collect();
        components.push(nodes);
        let mut distributions: Vec<FxHashSet<DistributionIndex>> = vec![];
        let ds = (0..g.number_distributions())
            .map(DistributionIndex)
            .collect::<FxHashSet<DistributionIndex>>();
        distributions.push(ds);
        Self {
            components,
            distributions,
        }
    }

    fn detect_components(
        &mut self,
        g: &Graph,
        state: &mut StateManager,
        _component: ComponentIndex,
    ) {
        let mut nodes: Vec<NodeIndex> = vec![];
        let mut distributions: FxHashSet<DistributionIndex> = FxHashSet::default();
        for n in g.nodes_iter() {
            if !g.is_node_bound(n, state) {
                nodes.push(n);
                if g.is_node_probabilistic(n) {
                    distributions.insert(g.get_distribution(n).unwrap());
                }
            }
        }
        self.components.push(nodes);
        self.distributions.push(distributions);
    }

    fn get_component_hash(&self, _component: ComponentIndex) -> u64 {
        0_u64
    }

    fn get_component_distributions(
        &self,
        component: ComponentIndex,
    ) -> &FxHashSet<DistributionIndex> {
        &self.distributions[component.0]
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
mod test_dfs_component {
    use super::{ComponentExtractor, ComponentIndex};
    use crate::core::graph::{DistributionIndex, Graph, NodeIndex};
    use crate::core::trail::{IntManager, SaveAndRestore, StateManager};
    use rustc_hash::FxHashSet;

    #[test]
    fn test_initialization_extractor() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3], n[4]], &mut state);

        let mut component_extractor = ComponentExtractor::new(&g, &mut state);
        component_extractor.detect_components(&g, &mut state, ComponentIndex(0));
        assert_eq!(5, component_extractor.nodes.len());
        assert_eq!(5, component_extractor.positions.len());
        assert_eq!(0, component_extractor.distributions[0].len());
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
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3], n[4]], &mut state);

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
        let expected = n.iter().copied().collect::<FxHashSet<NodeIndex>>();
        assert_eq!(expected, nodes);
    }

    #[test]
    fn test_initialization_multiple_components() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![n[1]], &mut state);
        g.add_clause(n[2], &vec![n[3], n[4]], &mut state);

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
        let n1 = vec![n[0], n[1]]
            .iter()
            .copied()
            .collect::<FxHashSet<NodeIndex>>();
        let n2 = vec![n[2], n[3], n[4]]
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
    }

    #[test]
    fn test_breaking_components() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3], n[4]], &mut state);

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
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3], n[4]], &mut state);

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
        let n = (0..2)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        let d1 = g.add_distribution(&vec![0.3, 0.3, 0.4], &mut state);
        let d2 = g.add_distribution(&vec![0.5, 0.5], &mut state);
        let d3 = g.add_distribution(&vec![0.3, 0.7], &mut state);

        g.add_clause(n[0], &d1, &mut state);
        g.add_clause(d2[0], &vec![n[0], n[1]], &mut state);
        g.add_clause(n[1], &d3, &mut state);

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
        assert_eq!(2, components.len());
        let distribution_comp1 = component_extractor.get_component_distributions(components[0]);
        assert_eq!(1, distribution_comp1.len());
        assert!(distribution_comp1.contains(&DistributionIndex(0)));
        let distribution_comp2 = component_extractor.get_component_distributions(components[1]);
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

    #[test]
    fn test_one_component() {
        let mut state = StateManager::default();
        let mut g = Graph::new();
        let n = (0..6)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[4], &vec![n[0]], &mut state);
        g.add_clause(n[1], &vec![n[0]], &mut state);
        g.add_clause(n[1], &vec![n[4]], &mut state);
        g.add_clause(n[3], &vec![n[4]], &mut state);
        g.add_clause(n[2], &vec![n[1]], &mut state);
        g.add_clause(n[3], &vec![n[2]], &mut state);
        g.add_clause(n[5], &vec![n[3]], &mut state);
        let mut component_extractor = ComponentExtractor::new(&g, &mut state);
        component_extractor.detect_components(&g, &mut state, ComponentIndex(0));
        assert_float_relative_eq!(
            0.415,
            component_extractor.fiedler_score[component_extractor.positions[0]],
            0.01
        );
        assert_float_relative_eq!(
            0.309,
            component_extractor.fiedler_score[component_extractor.positions[1]],
            0.01
        );
        assert_float_relative_eq!(
            0.069,
            component_extractor.fiedler_score[component_extractor.positions[2]],
            0.01
        );
        assert_float_relative_eq!(
            -0.221,
            component_extractor.fiedler_score[component_extractor.positions[3]],
            0.01
        );
        assert_float_relative_eq!(
            0.221,
            component_extractor.fiedler_score[component_extractor.positions[4]],
            0.01
        );
        assert_float_relative_eq!(
            -0.794,
            component_extractor.fiedler_score[component_extractor.positions[5]],
            0.01
        );
    }
}
