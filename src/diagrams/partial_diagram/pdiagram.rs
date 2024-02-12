//Schlandals
//Copyright (C) 2022 A. Dubray, L. Dierckx
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

/// Structure used to represent a partial search tree while solving a problem.
/// It is used when the search space is explored iteratively as it ease the storage of informations
/// about the solved sub-problems.
/// It should be noted that it is quite different than the arithmetic circuits.
/// Indeed, the goal of this struct is not to be evaluated multiple times for learning. Moreover,
/// it keeps the structure of the search algorithm (i.e., it does not have leaves representing
/// distributions).
use crate::core::graph::{VariableIndex, DistributionIndex};
use super::node::Node;
use crate::diagrams::NodeIndex;

use rug::Float;

#[derive(Clone, Copy)]
pub enum Child {
    Unexplored,
    OrChild(NodeIndex),
    AndChild(NodeIndex, VariableIndex),
}

pub struct PDiagram {
    nodes: Vec<Node>,
    children: Vec<Child>,
}

impl PDiagram {

    pub fn new() -> Self {
        Self {
            nodes: vec![],
            children: vec![],
        }
    }

    /// Adds a new OR node to the diagram and returns its index
    pub fn add_or_node(&mut self, decision: DistributionIndex, number_children: usize, max_proba: Float) -> NodeIndex {
        let index = NodeIndex(self.nodes.len());
        let child_start = self.children.len();
        self.nodes.push(Node::or_node(decision, child_start, number_children, max_proba));
        self.children.resize(self.children.len() + number_children, Child::Unexplored);
        index
    }

    /// Adds a new AND node to the diagram and returns its index
    pub fn add_and_node(&mut self, max_proba: Float, mut children: Vec<NodeIndex>) -> NodeIndex {
        let index = NodeIndex(self.nodes.len());
        let child_start = self.children.len();
        self.nodes.push(Node::and_node(max_proba, child_start, children.len()));
        while let Some(child) = children.pop() {
            self.children.push(Child::OrChild(child));
        }
        index
    }

    /// Returns the child at the given index
    pub fn get_child_at(&self, index: usize) -> Child {
        self.children[index]
    }

    pub fn number_nodes(&self) -> usize {
        self.nodes.len()
    }
}

impl std::ops::Index<NodeIndex> for PDiagram {
    type Output = Node;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl std::ops::IndexMut<NodeIndex> for PDiagram {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        &mut self.nodes[index.0]
    }
}
