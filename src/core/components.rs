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

#![allow(dead_code)]
use super::graph::{DistributionIndex, Graph, NodeIndex};
use super::trail::{ReversibleInt, StateManager};
use rustc_hash::{FxHashSet, FxHasher};
use std::hash::Hasher;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ComponentIndex(pub usize);

pub trait ComponentExtractor {
    fn detect_components<S: StateManager>(
        &mut self,
        g: &Graph,
        state: &mut S,
        component: ComponentIndex,
    );
    fn get_component(&self, component: ComponentIndex) -> &[NodeIndex];
    fn get_component_hash(&self, component: ComponentIndex) -> u64;
    fn get_component_distributions(
        &self,
        component: ComponentIndex,
    ) -> &FxHashSet<DistributionIndex>;
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

pub struct DFSComponentExtractor {
    nodes: Vec<NodeIndex>,
    positions: Vec<usize>,
    /// Holds the components computed by the extractor during the search
    components: Vec<Component>,
    distributions: Vec<FxHashSet<DistributionIndex>>,
    base: ReversibleInt,
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
        if g.is_node_bound(node, state)
            || comp_start <= node_pos && node_pos < (comp_start + *comp_size)
        {
            return;
        }
        // The new component of this node
        if !g.is_node_deterministic(node) {
            let distribution = g.get_distribution(node).unwrap();
            self.distributions.last_mut().unwrap().insert(distribution);
        }
        // Swap the NodeIndex in the range of the component in the vector
        // Current position of `node` in the vector `nodes`
        let current_pos = self.positions[node.0];
        let new_pos = comp_start + *comp_size;
        let moved_node = self.nodes[new_pos];
        if new_pos != current_pos {
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
                self.explore_component(g, node, start, &mut comp_size, state);
                self.components.push(Component(start, comp_size));
                let ns = &self.nodes;
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
