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
//!     3. u and v are not bound
//!
//! During the search, the propagators assign values to nodes and deactivate edges. The extractors
//! must implement the `ComponentExtractor` trait and compute the connected components when
//! `find_components` is called. This can be done incrementally or not.

use super::graph::{DistributionIndex, Graph, NodeIndex};
use super::trail::{ReversibleInt, StateManager};
use rustc_hash::{FxHashSet, FxHasher};
use std::hash::Hasher;

/// Abstraction used as a typesafe way of retrieving a `Component`
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ComponentIndex(pub usize);

/// This trait describes how components extractor must be implemented. An extractor of components
/// is a data structure that can, during the search, gives a list of the connected components (as
/// described above) in the graph
pub trait ComponentExtractor {
    /// This function is responsible of updating the data structure with the new connected
    /// components in `g` gien its current assignments.
    fn detect_components<S: StateManager>(
        &mut self,
        g: &Graph,
        state: &mut S,
        component: ComponentIndex,
    );
    /// Returns the nodes of a given component
    fn get_component(&self, component: ComponentIndex) -> &[NodeIndex];
    /// Returns the hash of a given component. This is the job of the implementing data structure
    /// to ensure that the same component gives the same hash, even if it appears in another part
    /// of the search tree
    fn get_component_hash(&self, component: ComponentIndex) -> u64;
    /// Returns the distributions in a given component
    fn get_component_distributions(
        &self,
        component: ComponentIndex,
    ) -> &FxHashSet<DistributionIndex>;
    /// Returns an iterator over the current components
    fn components_iter<S: StateManager>(&self, state: &S) -> ComponentIterator;
    /// Returns the current number of component
    fn number_components<S: StateManager>(&self, state: &S) -> usize;
}

/// A Component is identified by two integers. The first is the index in the vector of nodes at
/// which the component starts, and the second is the size of the component.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Component(pub usize, pub usize);

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
/// extract sub-components. It works by keeping all the `NodeIndex` in a single vector (the `nodes`
/// vector) so that all the nodes of a component are in a contiguous part of the vectors.
/// Thus in that case a component is simply two integers
///     1. The index of the first node in the component
///     2. The size of the component
///
/// Since the DFS is done using the `Graph` structure (i.e. following the edges of a node), a
/// second vector for the position of each node is kept.
/// This second vector is also used to know if a node has been processed or not. If the current
/// component starts at index 0 and has already 4 elements in it, a node which position is 2 has
/// already been processed during the DFS and can be safely ignored.
/// Finally in order to be safe during the search and more specifically the backtrack, the nodes in
/// `nodes` are moved inside the component.
/// For instance let us assume that we have two components, the `nodes` vector might looks like
/// [0, 2, 4, 1 | 5, 7 ] (the | is the separation between the two components, and integers
/// represent the nodes).
/// If now the first component is split in two, the nodes are moved in the sub-vectors spanning the
/// first four indexes. Thus (assuming 0 and 1 are in the same component and 2 and 4 in another),
/// this is a valid representation of the new vector [0, 1 | 2, 4 | 5, 7] while this is not
/// [2, 4 | 5, 7 | 0, 1]
pub struct DFSComponentExtractor {
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
}

impl DFSComponentExtractor {
    pub fn new<S: StateManager>(g: &Graph, state: &mut S) -> Self {
        let nodes = (0..g.number_nodes()).map(|i| NodeIndex(i)).collect();
        let positions = (0..g.number_nodes()).collect();
        let components = vec![Component(0, g.number_nodes())];
        let first_distributions = (0..g.number_distributions())
            .map(|i| DistributionIndex(i))
            .collect::<FxHashSet<DistributionIndex>>();
        let mut extractor = Self {
            nodes,
            positions,
            components,
            distributions: vec![first_distributions],
            base: state.manage_int(0),
            limit: state.manage_int(1),
        };
        extractor.detect_components(g, state, ComponentIndex(0));
        extractor
    }

    fn explore_component<S: StateManager>(
        &mut self,
        g: &Graph,
        node: NodeIndex,
        comp_start: usize,
        comp_size: &mut usize,
        state: &mut S,
    ) {
        let node_pos = self.positions[node.0];
        if !g.is_node_bound(node, state)
            && !(comp_start <= node_pos && node_pos < (comp_start + *comp_size))
        {
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

            // If the node is probabilistic, add its distribution to the set of distribution for this
            // component
            // And we need to add the nodes in the distribution in the component
            if g.is_node_probabilistic(node) {
                let distribution = g.get_distribution(node).unwrap();
                self.distributions.last_mut().unwrap().insert(distribution);
                for n in g.distribution_iter(distribution) {
                    if n != node {
                        self.explore_component(g, n, comp_start, comp_size, state);
                    }
                }
            }

            // Recursively explore the nodes in the connected components (i.e. linked by a clause)
            for clause in g.node_clauses(node) {
                for edge in g.edges_clause(clause) {
                    if g.is_edge_active(edge, state) {
                        let src = g.get_edge_source(edge);
                        let dst = g.get_edge_destination(edge);
                        self.explore_component(g, src, comp_start, comp_size, state);
                        self.explore_component(g, dst, comp_start, comp_size, state);
                    }
                }
            }
        }
    }
}

impl ComponentExtractor for DFSComponentExtractor {
    fn detect_components<S: StateManager>(
        &mut self,
        g: &Graph,
        state: &mut S,
        component: ComponentIndex,
    ) {
        let end = state.get_int(self.limit);
        self.components.truncate(end as usize);
        self.distributions.truncate(end as usize);
        state.set_int(self.base, end);
        let comp = self.components[component.0];
        let mut start = comp.0;
        let end = start + comp.1;
        while start < end {
            let node = self.nodes[start];
            if !g.is_node_bound(node, state) {
                let mut comp_size = 0;
                self.distributions.push(FxHashSet::default());
                self.explore_component(g, node, start, &mut comp_size, state);
                self.components.push(Component(start, comp_size));
                // Here we sort the nodes in the `nodes` vector to ensure that the hash of the
                // component is the same everywhere in the search tree. This is due to the fact
                // that hashing the same sequence of bytes in a different order will result in a
                // different hash.
                // This is probably not ideal but as a first solution it is fine.
                // We could add in each component a pointer to the node with the lowest index and
                // start a DFS on the graph from that. This should give a unique hash to the same
                // component.
                self.nodes[start..(start + comp_size)].sort();
                for i in start..start + comp_size {
                    self.positions[self.nodes[i].0] = i;
                }
                start += comp_size;
            } else {
                start += 1;
            }
        }
        state.set_int(self.limit, self.components.len() as isize);
    }

    /// Returns the nodes in a given component
    fn get_component(&self, component: ComponentIndex) -> &[NodeIndex] {
        let c = self.components[component.0];
        &self.nodes[c.0..(c.0 + c.1)]
    }

    /// Returns the hash of a component
    fn get_component_hash(&self, component: ComponentIndex) -> u64 {
        let mut hasher = FxHasher::default();
        let comp = self.components[component.0];
        for node in &self.nodes[comp.0..(comp.0 + comp.1)] {
            hasher.write_usize(node.0);
        }
        hasher.finish()
    }

    /// Returns the distributions in a component
    fn get_component_distributions(
        &self,
        component: ComponentIndex,
    ) -> &FxHashSet<DistributionIndex> {
        &self.distributions[component.0]
    }

    /// Returns an iterator over the node in a component
    fn components_iter<S: StateManager>(&self, state: &S) -> ComponentIterator {
        let start = state.get_int(self.base) as usize;
        let limit = state.get_int(self.limit) as usize;
        ComponentIterator { limit, next: start }
    }

    /// Returns the number of components detected by the last call of `detect_components`
    fn number_components<S: StateManager>(&self, state: &S) -> usize {
        (state.get_int(self.limit) - state.get_int(self.base)) as usize
    }
}

/// This structure is used to implement a simple component detector that always returns one
/// component with all the unassigned node in it. It is used to isolate bugs annd should not be
/// used for real data sets (as it performences will be terrible)
pub struct NoComponentExtractor {
    components: Vec<Vec<NodeIndex>>,
    distributions: Vec<FxHashSet<DistributionIndex>>,
}

impl NoComponentExtractor {
    pub fn new(g: &Graph) -> Self {
        let mut components: Vec<Vec<NodeIndex>> = vec![];
        let nodes = g.nodes_iter().collect();
        components.push(nodes);
        let mut distributions: Vec<FxHashSet<DistributionIndex>> = vec![];
        let ds = (0..g.number_distributions())
            .map(|i| DistributionIndex(i))
            .collect::<FxHashSet<DistributionIndex>>();
        distributions.push(ds);
        Self {
            components,
            distributions,
        }
    }
}

impl ComponentExtractor for NoComponentExtractor {
    fn detect_components<S: StateManager>(
        &mut self,
        g: &Graph,
        state: &mut S,
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

    fn get_component(&self, component: ComponentIndex) -> &[NodeIndex] {
        &self.components[component.0]
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

    fn components_iter<S: StateManager>(&self, _state: &S) -> ComponentIterator {
        ComponentIterator {
            limit: self.components.len(),
            next: self.components.len() - 1,
        }
    }

    fn number_components<S: StateManager>(&self, _state: &S) -> usize {
        1
    }
}

#[cfg(test)]
mod test_dfs_component {
    use super::{ComponentExtractor, ComponentIndex, DFSComponentExtractor};
    use crate::core::graph::{DistributionIndex, Graph, NodeIndex};
    use crate::core::trail::{IntManager, SaveAndRestore, TrailedStateManager};
    use rustc_hash::{FxHashSet, FxHasher};
    use std::hash::Hasher;

    #[test]
    fn test_initialiaztion_extractor() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new();
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3], n[4]], &mut state);

        let component_extractor = DFSComponentExtractor::new(&g, &mut state);
        assert_eq!(5, component_extractor.nodes.len());
        assert_eq!(5, component_extractor.positions.len());
        assert_eq!(0, component_extractor.distributions[0].len());
        assert_eq!(1, state.get_int(component_extractor.base));
        assert_eq!(2, state.get_int(component_extractor.limit));
        assert_eq!(1, component_extractor.number_components(&state));
    }

    #[test]
    fn test_initialization_extractor_distribution() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new();
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        let d = g.add_distribution(&vec![0.3, 0.4, 0.3], &mut state);
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3], n[4]], &mut state);
        g.add_clause(n[1], &vec![d[0]], &mut state);
        g.add_clause(n[3], &vec![d[1], d[2]], &mut state);
        let component_extractor = DFSComponentExtractor::new(&g, &mut state);
        assert_eq!(8, component_extractor.nodes.len());
        assert_eq!(8, component_extractor.positions.len());
        assert_eq!(1, component_extractor.distributions[0].len());
        assert_eq!(1, state.get_int(component_extractor.base));
        assert_eq!(2, state.get_int(component_extractor.limit));
        assert_eq!(1, component_extractor.number_components(&state));
    }

    #[test]
    fn test_initialization_single_component() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new();
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3], n[4]], &mut state);

        let component_extractor = DFSComponentExtractor::new(&g, &mut state);
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(1, components.len());
        assert_eq!(1, component_extractor.number_components(&state));
        let nodes = component_extractor
            .get_component(components[0])
            .iter()
            .copied()
            .collect::<FxHashSet<NodeIndex>>();
        let expected = n.iter().copied().collect::<FxHashSet<NodeIndex>>();
        assert_eq!(expected, nodes);
    }

    #[test]
    fn test_initialization_multiple_components() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new();
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![n[1]], &mut state);
        g.add_clause(n[2], &vec![n[3], n[4]], &mut state);

        let component_extractor = DFSComponentExtractor::new(&g, &mut state);
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(2, components.len());
        assert_eq!(2, component_extractor.number_components(&state));
        let n_c1 = component_extractor
            .get_component(components[0])
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
        let n_c2 = component_extractor
            .get_component(components[1])
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
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new();
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3], n[4]], &mut state);

        let mut component_extractor = DFSComponentExtractor::new(&g, &mut state);
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(1, components.len());
        assert_eq!(1, component_extractor.number_components(&state));

        g.set_node(n[1], true, &mut state);
        component_extractor.detect_components(&g, &mut state, components[0]);
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(2, component_extractor.number_components(&state));
        assert_eq!(2, components.len());
    }

    #[test]
    fn test_breaking_component_but_backtrack() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new();
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3], n[4]], &mut state);

        let mut component_extractor = DFSComponentExtractor::new(&g, &mut state);
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(1, components.len());
        assert_eq!(1, component_extractor.number_components(&state));

        state.save_state();

        g.set_node(n[1], true, &mut state);
        component_extractor.detect_components(&g, &mut state, components[0]);
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(2, components.len());
        assert_eq!(2, component_extractor.number_components(&state));

        state.restore_state();
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(1, components.len());
        assert_eq!(1, component_extractor.number_components(&state));
    }

    #[test]
    fn test_hash() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new();
        let n = (0..5)
            .map(|_| g.add_node(false, None, None, &mut state))
            .collect::<Vec<NodeIndex>>();
        g.add_clause(n[0], &vec![n[1], n[2]], &mut state);
        g.add_clause(n[1], &vec![n[3], n[4]], &mut state);
        let mut component_extractor = DFSComponentExtractor::new(&g, &mut state);
        let components: Vec<ComponentIndex> = component_extractor.components_iter(&state).collect();
        assert_eq!(1, components.len());

        let comp = components[0];
        let mut hasher = FxHasher::default();
        for i in 0..5 {
            hasher.write_usize(i);
        }
        assert_eq!(
            hasher.finish(),
            component_extractor.get_component_hash(comp)
        );

        g.set_node(n[1], false, &mut state);

        component_extractor.detect_components(&g, &mut state, comp);
        let components: Vec<ComponentIndex> = component_extractor.components_iter(&state).collect();
        assert_eq!(2, components.len());
        let mut hasher_comp1 = FxHasher::default();
        hasher_comp1.write_usize(0);
        hasher_comp1.write_usize(2);
        assert_eq!(
            hasher_comp1.finish(),
            component_extractor.get_component_hash(components[0])
        );

        let mut hasher_comp2 = FxHasher::default();
        hasher_comp2.write_usize(3);
        hasher_comp2.write_usize(4);
        assert_eq!(
            hasher_comp2.finish(),
            component_extractor.get_component_hash(components[1])
        );
    }

    #[test]
    fn distributions_in_components() {
        let mut state = TrailedStateManager::new();
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

        let mut component_extractor = DFSComponentExtractor::new(&g, &mut state);
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
        assert_eq!(2, distribution_comp1.len());
        assert!(distribution_comp1.contains(&DistributionIndex(1)));
        assert!(distribution_comp1.contains(&DistributionIndex(2)));
        let distribution_comp2 = component_extractor.get_component_distributions(components[1]);
        assert_eq!(1, distribution_comp2.len());
        assert!(distribution_comp2.contains(&DistributionIndex(0)));
    }
}
