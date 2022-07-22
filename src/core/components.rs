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
}

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
        // If the node is bound, it is not part of any component so we can return. The other
        // condition states that if the position of the node is between the start of the component
        // being build, and the last node that was added in the component, then it has already
        // been processed.
        if g.is_node_bound(node, state)
            || comp_start <= node_pos && node_pos < (comp_start + *comp_size)
        {
            return;
        }

        // If the node is probabilistic, add its distribution to the set of distribution for this
        // component
        if !g.is_node_deterministic(node) {
            let distribution = g.get_distribution(node).unwrap();
            self.distributions.last_mut().unwrap().insert(distribution);
        }

        // This effectively add the node in the component.
        let current_pos = self.positions[node.0];
        // The node is placed at the end of the component
        let new_pos = comp_start + *comp_size;
        if new_pos != current_pos {
            let moved_node = self.nodes[new_pos];
            self.nodes.as_mut_slice().swap(new_pos, current_pos);
            self.positions[node.0] = new_pos;
            self.positions[moved_node.0] = current_pos;
        }
        *comp_size += 1;
        // Iterates over all the incoming edges and recursively visit the parents if the
        // edge is active.
        // Unfortunately we can not filter the edges in the iterator because the state need
        // to be passed to the `is_edge_active` function and thus would be immutably and
        // mutably borrowed at the same time
        for edge in g.incomings(node) {
            if g.is_edge_active(edge, state) {
                let src = g.get_edge_source(edge);
                self.explore_component(g, src, comp_start, comp_size, state);
            }
        }
        for edge in g.outgoings(node) {
            if g.is_edge_active(edge, state) {
                let dst = g.get_edge_destination(edge);
                self.explore_component(g, dst, comp_start, comp_size, state);
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
                let ns = &self.nodes;
                // Here we sort the nodes in the `nodes` vector to ensure that the hash of the
                // component is the same everywhere in the search tree. This is due to the fact
                // that hashing the same sequence of bytes in a different order will result in a
                // different hash.
                // This is probably not ideal but as a first solution it is fine.
                // We could add in each component a pointer to the node with the lowest index and
                // start a DFS on the graph from that. This should give a unique hash to the same
                // component.
                self.positions[start..(start + comp_size)].sort_by(|n1, n2| ns[*n1].cmp(&ns[*n2]));
                self.nodes[start..(start + comp_size)].sort();
                start += comp_size;
            } else {
                start += 1;
            }
        }
        state.set_int(self.limit, self.components.len() as isize);
    }

    fn get_component(&self, component: ComponentIndex) -> &[NodeIndex] {
        let c = self.components[component.0];
        &self.nodes[c.0..(c.0 + c.1)]
    }

    fn get_component_hash(&self, component: ComponentIndex) -> u64 {
        let mut hasher = FxHasher::default();
        let comp = self.components[component.0];
        for node in &self.nodes[comp.0..(comp.0 + comp.1)] {
            hasher.write_usize(node.0);
        }
        hasher.finish()
    }

    fn get_component_distributions(
        &self,
        component: ComponentIndex,
    ) -> &FxHashSet<DistributionIndex> {
        &self.distributions[component.0]
    }

    fn components_iter<S: StateManager>(&self, state: &S) -> ComponentIterator {
        let start = state.get_int(self.base) as usize;
        let limit = state.get_int(self.limit) as usize;
        ComponentIterator { limit, next: start }
    }
}

#[cfg(test)]
mod test_dfs_component {
    use super::{ComponentExtractor, ComponentIndex, DFSComponentExtractor};
    use crate::core::graph::{Graph, NodeIndex};
    use crate::core::trail::{SaveAndRestore, TrailedStateManager};
    use rustc_hash::FxHashSet;

    #[test]
    fn test_initialization_single_component() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
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
        let mut g = Graph::new(&mut state);
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
        let mut g = Graph::new(&mut state);
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

        g.set_node(n[1], true, &mut state);
        component_extractor.detect_components(&g, &mut state, components[0]);
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(2, components.len());
    }

    #[test]
    fn test_breaking_component_but_backtrack() {
        let mut state = TrailedStateManager::new();
        let mut g = Graph::new(&mut state);
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

        state.save_state();

        g.set_node(n[1], true, &mut state);
        component_extractor.detect_components(&g, &mut state, components[0]);
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(2, components.len());

        state.restore_state();
        let components = component_extractor
            .components_iter(&state)
            .collect::<Vec<ComponentIndex>>();
        assert_eq!(1, components.len());
    }
}
