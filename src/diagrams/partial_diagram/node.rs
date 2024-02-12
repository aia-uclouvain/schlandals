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

use crate::core::graph::DistributionIndex;
use crate::solvers::Bounds;
use crate::diagrams::*;
use crate::common::f128;
use rug::{Assign, Float};

pub struct Node {
    /// If the node is an OR Node (or Sum node), this is the distribution that is branched on
    decision: Option<DistributionIndex>,
    /// The accumulated probabilities of (non-)models of the sub-problem being soled at that node
    bounds: Bounds,
    /// First index of the children of the node
    child_start: usize,
    /// Number of children the node has
    number_children: usize,
    /// Maximum probability that can be obtained in this node. That is, this is the probability
    /// that would be returned if all remaining interpretation would models
    max_proba: Float,
    /// Indicates if the node is sat, unsat or not finished to be explored
    status: NodeStatus,
}

impl Node {

    /// Returns a new OR node
    pub fn or_node(decision: DistributionIndex, child_start: usize, number_children: usize, max_proba: Float) -> Self {
        Self {
            decision: Some(decision),
            bounds: (f128!(0.0), f128!(0.0)),
            child_start,
            number_children,
            max_proba,
            status: NodeStatus::Unknown,
        }
    }

    /// Returns a new AND node
    pub fn and_node(max_proba: Float, child_start: usize, number_children: usize) -> Self {
        Self {
            decision: None,
            bounds: (f128!(0.0), f128!(0.0)),
            child_start,
            number_children,
            max_proba,
            status: NodeStatus::Unknown,
        }
    }

    pub fn maximum_probability(&self) -> &Float {
        &self.max_proba
    }

    /// Returns true if the node has a decision
    pub fn decision(&self) -> Option<DistributionIndex> {
        self.decision
    }

    /// Sets the decision for this node
    pub fn set_decision(&mut self, decision: Option<DistributionIndex>) {
        self.decision = decision;
    }

    /// Returns an iterator on the ids of the node's children
    pub fn children_iter(&self) -> impl Iterator<Item = usize> {
        self.child_start..(self.child_start + self.number_children)
    }

    pub fn child_start(&self) -> usize {
        self.child_start
    }

    pub fn is_sat(&self) -> bool {
        is_node_type!(self.status, NodeStatus::Sat)
    }

    pub fn set_sat(&mut self) {
        self.status = NodeStatus::Sat;
    }

    pub fn is_unsat(&self) -> bool {
        is_node_type!(self.status, NodeStatus::Unsat)
    }

    pub fn set_unsat(&mut self) {
        self.bounds.0.assign(0.0);
        self.bounds.1.assign(&self.max_proba);
        self.status = NodeStatus::Unsat;
    }

    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    pub fn set_bounds(&mut self, bounds: Bounds) {
        self.bounds = bounds;
    }
}
