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

//! This module implements a graph using vector indexes. This implementation is inspired by [this
//! blog post](https://smallcultfollowing.com/babysteps/blog/2015/04/06/modeling-graphs-in-rust-using-vector-indices/)
//! and [this
//! code](https://github.com/xgillard/ddo/blob/master/src/implementation/mdd/deep/mddgraph.rs).
//!
//! A graph G = (V, E) is represented using two vectors containing, respectively, all the nodes and
//! all the edges. A node (or an edge) is identified uniquely by its index in its enclosing vector.
//! The `VariableIndex` and `EdgeIndex` structures are used to index the nodes and edges.
//!
//! The parents and children of each node is implemented as a succession of 'pointers' of
//! `EdgeIndex`.
//! If a node n1 has two children n2, n3 then there are two directed edges in the graph n1 -> n2
//! and n1 -> n3.
//! These edges are respectively indexed by e1 and e2.
//! In the `Variable` structure, the field `children` is filled with the value `Some(e1)`, which
//! references the first of its outgoing edges (to its children n2).
//! In the `Edge` for the edge e1, the field `next_outgoing` is set to `Some(e2)`, the second
//! outgoing edge of n1.
//! On the other hand, since there are no more child to n1 after n3, this field is `None` for the
//! edge identified by e2.
//!
//! # Note:
//! Once the graph is constructed, no edge/node should be removed from it. Thus this
//! implementation does not have problems like dangling indexes.

use search_trail::*;
use rustc_hash::FxHashMap;

// The following abstractions allow to have type safe indexing for the nodes, edes and clauses.
// They are used to retrieve respectively `Variable`, `Edge` and `Clause` in the `Graph`
// structure.

/// Abstraction used as a typesafe way of retrieving a `Variable` in the `Graph` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VariableIndex(pub usize);

/// Abstraction used as a typesafe way of retrieving a `Clause` in the `Graph` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ClauseIndex(pub usize);

/// Abstraction used as a typesafe way of retrieving a `Distribution` in the `Graph` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DistributionIndex(pub usize);

/// Data structure that actually holds the data of a  node in the graph
/// A node in the graph is in four possible states: 1) Unassigned 2) True 3) False 4)
/// Unconstrained
///
/// In the last case, it means that the node can be either `true` or `false` without impacting the
/// counting.
/// The value of the node is stored in the `value` field and its domain is implicitly given by the
/// `domain_size` field.
/// If a `domain_size = 2` then both `true` and `false` are in the domain, and the variable is
/// unassgined. If `domain_size = 1` then the value is assigned to the value in the `value` field.
///
/// # Note:
/// This might not be the best design, but it seems that a full handling of domain etc (like in
/// a cp solver) is a bit overkill since at the moment we only need BoolVar.
#[derive(Debug, Clone)]
pub struct Variable {
    /// Indicate if the literal represented by the node is a probabilistic literal (i.e. have a
    /// weight) or not
    pub probabilistic: bool,
    /// If `probabilistic` is `true`, then this is the weight associated to the node. Otherwise
    /// this is None. The weight is assumed to be log-transformed.
    pub weight: Option<f64>,
    /// If `probabilistic` is `true`, this is the index of the distribution containing this node
    pub distribution: Option<DistributionIndex>,
    /// The clauses in which the variable appears as the head
    pub clauses_head: Vec<ClauseIndex>,
    /// The clauses in which the variable appears in the body
    pub clauses_body: Vec<ClauseIndex>,
    /// The value assigned to the variable
    value: bool,
    /// True if the variable is assigned
    bound: ReversibleBool,
    /// Number of active clause containing the variable
    number_active_clause: ReversibleUsize,
}

/// This structure represent a clause in the graph. A clause is of the form
///     a && b && ... && d => e
/// In the graph, this will be represented by n (the number of literals in the implicant) incoming
/// edges in `e`.
/// The edges of a clause are added at the same time, so a clause can be fully identified by an
/// `EdgeIndex` (the first edge, in the example `a -> e`) and the size of the clause (n)
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Clause {
    /// Head of the clause
    pub head: VariableIndex,
    /// Body of the clause, cunjunction of all the variable in the body
    pub body: Vec<VariableIndex>,
    /// Is the clause constrained? A clause is unconstrained if it always evaluates to T, independently of its unassigned var
    pub constrained: ReversibleBool,
    /// Numbers of probabilistic variable active in the clause body
    pub number_probabilistic: ReversibleUsize,
    /// Numbers of deterministic variable active in the clause body
    pub number_deterministic: ReversibleUsize,
    /// Link to the first children in the graph of clauses
    pub children: Vec<ClauseIndex>,
    /// Link to the first parent in the graph of clauses
    pub parents: Vec<ClauseIndex>,
    map_var_body_idx: FxHashMap<VariableIndex, usize>,
    body_size: ReversibleUsize,
    map_var_parent_idx: FxHashMap<ClauseIndex, usize>,
    number_parent: ReversibleUsize,
    map_var_child_idx: FxHashMap<ClauseIndex, usize>,
    number_child: ReversibleUsize,
}

/// Represents a set of nodes in a same distribution. This assume that the nodes of a distribution
/// are inserted in the graph one after the other (i.e. that their `VariableIndex` are consecutive).
/// Since no node should be removed from the graph once constructed, this should not be a problem.
/// Thus a distribution is identified by the first `VariableIndex` and the number of nodes in the
/// distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Distribution {
    /// First node in the distribution
    pub first: VariableIndex,
    /// Number of node in the distribution
    pub size: usize,
    /// Number of clauses in which the distribution appears
    pub number_clause_active: ReversibleUsize,
    /// Number of nodes assigned to F in the distribution
    pub number_false: ReversibleUsize,
}

/// Data structure representing the Graph.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Vector containing the nodes of the graph
    variables: Vec<Variable>,
    /// Vector containing the clauses of the graph
    clauses: Vec<Clause>,
    /// Vector containing the distributions of the graph
    distributions: Vec<Distribution>,
    /// Number of probabilistic nodes in the graph
    number_probabilistic: usize,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    
    // --- GRAPH CREATION --- //

    pub fn new() -> Self {
        Self {
            variables: vec![],
            clauses: vec![],
            distributions: vec![],
            number_probabilistic: 0,
        }
    }

    /// Add a distribution to the graph. In this case, a distribution is a set of probabilistic
    /// variable such that
    ///     - The sum of their weights sum up to 1.0
    ///     - Exatctly one of these node is true in a solution
    ///     - None of the node in the distribution is part of another distribution
    ///
    /// Each probabilstic node should be part of one distribution.
    pub fn add_distribution(
        &mut self,
        weights: &[f64],
        state: &mut StateManager,
    ) -> Vec<VariableIndex> {
        self.number_probabilistic += weights.len();
        let distribution = DistributionIndex(self.distributions.len());
        let variables: Vec<VariableIndex> = weights
            .iter()
            .map(|w| self.add_variable(true, Some(*w), Some(distribution), state))
            .collect();
        self.distributions.push(Distribution {
            first: variables[0],
            size: variables.len(),
            number_clause_active: state.manage_usize(0),
            number_false: state.manage_usize(0),
        });
        variables 
    }
    
    /// Adds a variable to the graph, which is used later by the clauses.
    /// Returns the index of the variables in the `variables` vector.
    pub fn add_variable(
        &mut self,
        probabilistic: bool,
        weight: Option<f64>,
        distribution: Option<DistributionIndex>,
        state: &mut StateManager,
    ) -> VariableIndex {
        let id = self.variables.len();
        self.variables.push(Variable {
            probabilistic,
            weight,
            distribution,
            clauses_head: vec![],
            clauses_body: vec![],
            value: false,
            bound: state.manage_bool(false),
            number_active_clause: state.manage_usize(0),
        });
        VariableIndex(id)
    }

    /// Add a clause to the graph. A clause is a expression of the form
    ///     n1 && n2 && ... && nn => head
    ///
    /// where head, n1, ..., nn are variable of the graph. ´head` is the head of the clause and `body`
    /// = vec![n1, ..., nn].
    pub fn add_clause(
        &mut self,
        head: VariableIndex,
        body: Vec<VariableIndex>,
        state: &mut StateManager,
    ) -> ClauseIndex {
        let cid = ClauseIndex(self.clauses.len());
        let number_probabilistic = body.iter().copied().filter(|v| self.is_variable_probabilistic(*v)).count();
        let number_deterministic = body.len() - number_probabilistic;
        let body_size = body.len();
        let mut map_var_body_idx: FxHashMap<VariableIndex, usize> = FxHashMap::default();
        for i in 0..body_size {
            map_var_body_idx.insert(body[i], i);
        }
        let map_var_parent_idx: FxHashMap<ClauseIndex, usize> = FxHashMap::default();
        let map_var_child_idx: FxHashMap<ClauseIndex, usize> = FxHashMap::default();
        self.clauses.push(Clause {
            head,
            body: body.clone(),
            constrained: state.manage_bool(true),
            number_probabilistic: state.manage_usize(number_probabilistic),
            number_deterministic: state.manage_usize(number_deterministic),
            children: vec![],
            parents: vec![],
            map_var_body_idx,
            body_size: state.manage_usize(body_size),
            map_var_parent_idx,
            number_parent: state.manage_usize(0),
            map_var_child_idx,
            number_child: state.manage_usize(0),
        });
        state.increment_usize(self.variables[head.0].number_active_clause);
        for n in body {
            if self.is_variable_probabilistic(n) {
                let distribution = self.variables[n.0].distribution.unwrap();
                state.increment_usize(self.distributions[distribution.0].number_clause_active);
            }
            state.increment_usize(self.variables[n.0].number_active_clause);
            for clause in self.variables[n.0].clauses_head.clone() {
                self.add_edge(clause, cid, state);
            }
            self.variables[n.0].clauses_body.push(cid);
        }
        for clause in self.variables[head.0].clauses_body.clone() {
            self.add_edge(cid, clause, state);
        }
        self.variables[head.0].clauses_head.push(cid);
        state.set_usize(self.clauses[cid.0].number_probabilistic, number_probabilistic);
        state.set_usize(self.clauses[cid.0].number_deterministic, number_deterministic);
        cid
    }

    /// Add an edge between the node identified by `src` to the node identified by `dst`. This
    /// method returns the index of the edge.
    fn add_edge(
        &mut self,
        src: ClauseIndex,
        dst: ClauseIndex,
        state: &mut StateManager,
    ) {
        let source_nb_child = state.get_usize(self.clauses[src.0].number_child);
        self.clauses[src.0].map_var_child_idx.insert(dst, source_nb_child);
        self.clauses[src.0].children.push(dst);
        state.increment_usize(self.clauses[src.0].number_child);

        let target_nb_parent = state.get_usize(self.clauses[dst.0].number_parent);
        self.clauses[dst.0].map_var_parent_idx.insert(src, target_nb_parent);
        self.clauses[dst.0].parents.push(src);
        state.increment_usize(self.clauses[dst.0].number_parent);
    }
    
    // --- GRAPH MODIFICATIONS --- //
    
    fn remove_child(&mut self, clause: ClauseIndex, child: ClauseIndex, state: &mut StateManager) {
        let current_index = *self.clauses[clause.0].map_var_child_idx.get(&child).unwrap();
        let new_index = state.get_usize(self.clauses[clause.0].number_child) - 1;
        let swapped_child = self.clauses[clause.0].children[new_index];
        self.clauses[clause.0].children.swap(current_index, new_index);
        self.clauses[clause.0].map_var_child_idx.insert(swapped_child, current_index);
        self.clauses[clause.0].map_var_child_idx.insert(child, new_index);
        state.decrement_usize(self.clauses[clause.0].number_child);
    }

    fn remove_parent(&mut self, clause: ClauseIndex, parent: ClauseIndex, state: &mut StateManager) {
        let current_index = *self.clauses[clause.0].map_var_parent_idx.get(&parent).unwrap();
        let new_index = state.get_usize(self.clauses[clause.0].number_parent) - 1;
        let swapped_parent = self.clauses[clause.0].parents[new_index];
        self.clauses[clause.0].parents.swap(current_index, new_index);
        self.clauses[clause.0].map_var_parent_idx.insert(swapped_parent, current_index);
        self.clauses[clause.0].map_var_parent_idx.insert(parent, new_index);
        state.decrement_usize(self.clauses[clause.0].number_parent);
    }
    
    fn remove_variable_from_body(&mut self, variable: VariableIndex, clause: ClauseIndex, state: &mut StateManager) {
        let current_index = *self.clauses[clause.0].map_var_body_idx.get(&variable).unwrap();
        let new_index = state.get_usize(self.clauses[clause.0].body_size) - 1;
        let swapped_var = self.clauses[clause.0].body[new_index];
        self.clauses[clause.0].body.swap(current_index, new_index);
        self.clauses[clause.0].map_var_body_idx.insert(swapped_var, current_index);
        self.clauses[clause.0].map_var_body_idx.insert(variable, new_index);
        state.decrement_usize(self.clauses[clause.0].body_size);
    }
    
    /// Sets a variable to true or false.
    ///     - If true, it is "removed" from the implicant of all the clauses in which it appears.
    ///     It also deactivate all the clauses that have the variable as head
    ///     - If false, it deactivate all the clauses in which it appears in the implicant. It also
    ///     make all the clauses in which it is the head bot-reachable
    pub fn set_variable(&mut self, variable: VariableIndex, value: bool, state: &mut StateManager) {
        state.set_bool(self.variables[variable.0].bound, true);
        self.variables[variable.0].value = value;
        if value {
            for i in 0..self.variables[variable.0].clauses_body.len() {
                let clause = self.variables[variable.0].clauses_body[i];
                if self.is_clause_constrained(clause, state) {
                    self.remove_variable_from_body(variable, clause, state);
                }
            }
        }
        if !value && self.is_variable_probabilistic(variable) {
            let d = self.get_variable_distribution(variable).unwrap();
            state.increment_usize(self.distributions[d.0].number_false);
        }
    }

    pub fn decrement_distribution_clause_counter(&self, distribution: DistributionIndex, state: &mut StateManager) -> usize {
        state.decrement_usize(self.distributions[distribution.0].number_clause_active)
    }
    
    /// Decrements the number of unassigned probabilistic variables in the body of a clause and returns the remaining
    /// number of unassigned variable in the body
    pub fn clause_decrement_number_probabilistic(&self, clause: ClauseIndex, state: &mut StateManager) -> usize {
        let remaining = state.get_usize(self.clauses[clause.0].number_deterministic);
        remaining + state.decrement_usize(self.clauses[clause.0].number_probabilistic)
    }

    /// Decrements the number of unassigned deterministic variables in the body of a clause and returns the remaining
    /// number of unassigned variable in the body
    pub fn clause_decrement_number_deterministic(&self, clause: ClauseIndex, state: &mut StateManager) -> usize {
        let remaining = state.get_usize(self.clauses[clause.0].number_probabilistic);
        remaining + state.decrement_usize(self.clauses[clause.0].number_deterministic)
    }
    
    // --- QUERIES --- //

    /// Returns true if the clause is bound. This happens if the size of the clause is 0
    pub fn is_clause_constrained(&self, clause: ClauseIndex, state: &StateManager) -> bool {
        state.get_bool(self.clauses[clause.0].constrained)
    }
    
    /// Returns true if the variable is bound
    pub fn is_variable_bound(&self, variable: VariableIndex, state: &StateManager) -> bool {
        state.get_bool(self.variables[variable.0].bound)
    } 
    
    /// Returns true if the variable if probabilistic
    pub fn is_variable_probabilistic(&self, variable: VariableIndex) -> bool {
        self.variables[variable.0].probabilistic
    }
    
    /// Returns the number of unassigned probabilistic variable in the clause body
    pub fn clause_number_probabilistic(&self, clause: ClauseIndex, state: &StateManager) -> usize {
        state.get_usize(self.clauses[clause.0].number_probabilistic)
    }

    /// Returns the number of unassigned deterministic variable in the clause body
    pub fn clause_number_deterministic(&self, clause: ClauseIndex, state: &StateManager) -> usize {
        state.get_usize(self.clauses[clause.0].number_deterministic)
    }
    
    /// Returns true if the clause still have probabilistic variable in its implicant
    pub fn clause_has_probabilistic(&self, clause: ClauseIndex, state: &StateManager) -> bool {
        self.clause_number_probabilistic(clause, state) != 0
    }
    
    /// Returns the number of unassigned variable in the body of a clause
    pub fn clause_number_unassigned(&self, clause: ClauseIndex, state: &StateManager) -> usize {
        state.get_usize(self.clauses[clause.0].number_deterministic) + state.get_usize(self.clauses[clause.0].number_probabilistic)
    }
    
    pub fn remove_clause_from_parent(&mut self, clause: ClauseIndex, state: &mut StateManager) {
        for parent in self.parents_clause_iter(clause, state).collect::<Vec<ClauseIndex>>() {
            self.remove_child(parent, clause, state);
        }
    }
    
    pub fn remove_clause_from_children(&mut self, clause: ClauseIndex, state: &mut StateManager) {
        for child in self.children_clause_iter(clause, state).collect::<Vec<ClauseIndex>>() {
            self.remove_parent(child, clause, state);
        }
    }

    /// Deactivate a clause
    pub fn set_clause_unconstrained(&self, clause: ClauseIndex, state: &mut StateManager) {
        state.set_bool(self.clauses[clause.0].constrained, false);
    }
    
    /// Returns true if there are only one variable unassigned in the distribution, and every other is F
    pub fn distribution_one_left(&self, distribution: DistributionIndex, state: &StateManager) -> bool {
        self.distributions[distribution.0].size - state.get_usize(self.distributions[distribution.0].number_false) == 1
    }
    
    /// Returns the number of false in the distribution
    pub fn distribution_number_false(&self, distribution: DistributionIndex, state: &StateManager) -> usize {
        state.get_usize(self.distributions[distribution.0].number_false)
    }
    
    // --- GETTERS --- //
    
    /// Returns the number of clause in the graph
    pub fn number_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Returns the number of variable in the graph
    pub fn number_variables(&self) -> usize {
        self.variables.len()
    }

    /// Returns the distribution of a probabilistic variable
    pub fn get_variable_distribution(&self, variable: VariableIndex) -> Option<DistributionIndex> {
        self.variables[variable.0].distribution
    }

    /// Returns the value of the vairable
    pub fn get_variable_value(&self, variable: VariableIndex, state: &StateManager) -> Option<bool> {
        if !self.is_variable_bound(variable, state) {
            None
        } else {
            Some(self.variables[variable.0].value)
        }
    }
    
    /// Returns the weight of the variable
    pub fn get_variable_weight(&self, variable: VariableIndex) -> Option<f64> {
        self.variables[variable.0].weight
    }
    
    /// Returns the head of the clause
    pub fn get_clause_head(&self, clause: ClauseIndex) -> VariableIndex {
        self.clauses[clause.0].head
    }
    
    /// Returns an active distribution in the clause body
    pub fn get_clause_active_distribution(&self, clause: ClauseIndex, state: &StateManager) -> Option<DistributionIndex> {
        self.clause_body_iter(clause, state).filter(|v| self.is_variable_probabilistic(*v) && !self.is_variable_bound(*v, state)).map(|v| self.get_variable_distribution(v).unwrap()).next()
    }

    // --- ITERATORS --- //
    
    /// Returns an iterator on the parents of a clause
    pub fn parents_clause_iter(&self, clause: ClauseIndex, state: &StateManager) -> impl Iterator<Item = ClauseIndex> + '_ {
        let size = state.get_usize(self.clauses[clause.0].number_parent);
        self.clauses[clause.0].parents[0..size].iter().copied()
    }
    
    /// Returns an iterator on the children of a clause
    pub fn children_clause_iter(&self, clause: ClauseIndex, state: &StateManager) -> impl Iterator<Item = ClauseIndex> + '_ {
        let size = state.get_usize(self.clauses[clause.0].number_child);
        self.clauses[clause.0].children[0..size].iter().copied()
    }
    
    /// Returns an iterator on the clauses of the graph
    pub fn clause_iter(&self) -> impl Iterator<Item = ClauseIndex> {
        (0..self.clauses.len()).map(ClauseIndex)
    }
    
    /// Returns an iterator over the clauses in which the variable is in the index
    pub fn variable_clause_body_iter(&self, variable: VariableIndex) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.variables[variable.0].clauses_body.iter().copied()
    }
    
    /// Returns an iterator over the clauses in which the variable is the head
    pub fn variable_clause_head_iter(&self, variable: VariableIndex) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.variables[variable.0].clauses_head.iter().copied()
    }

    /// Returns an iterator over the variable in a distribution
    pub fn distribution_variable_iter(&self, distribution: DistributionIndex) -> impl Iterator<Item = VariableIndex> {
        let first = self.distributions[distribution.0].first.0;
        let last = first + self.distributions[distribution.0].size;
        (first..last).map(VariableIndex)
    }

    /// Returns an iterator on the body of a clause
    pub fn clause_body_iter(&self, clause: ClauseIndex, state: &StateManager) -> impl Iterator<Item = VariableIndex> + '_ {
        let size = state.get_usize(self.clauses[clause.0].body_size);
        self.clauses[clause.0].body[0..size].iter().copied()
    }
    
}

#[cfg(test)]
mod test_graph_creation {
    
    use crate::core::graph::*;
    use search_trail::StateManager;

    #[test]
    pub fn variables_creation() {
        let mut g = Graph::default();
        let mut state = StateManager::default();
        let x = (0..3).map(|x| g.add_variable(true, Some(1.0 / (x + 1) as f64), Some(DistributionIndex(x)), &mut state)).collect::<Vec<VariableIndex>>();
        for i in 0..3 {
            assert_eq!(VariableIndex(i), x[i]);
            let v = &g.variables[x[i].0];
            let vec_clause: Vec<ClauseIndex> = vec![];
            assert!(v.probabilistic);
            assert_eq!(Some(1.0 / (i + 1) as f64), v.weight);
            assert_eq!(Some(DistributionIndex(i)), v.distribution);
            assert_eq!(vec_clause, v.clauses_head);
            assert_eq!(vec_clause, v.clauses_body);
            assert!(!state.get_bool(v.bound));
        }

        let x = (0..3).map(|_| g.add_variable(false, None, None, &mut state)).collect::<Vec<VariableIndex>>();
        for i in 0..3 {
            assert_eq!(VariableIndex(i+3), x[i]);
            let v = &g.variables[x[i].0];
            let vec_clauses: Vec<ClauseIndex> = vec![];
            assert!(!v.probabilistic);
            assert_eq!(None, v.weight);
            assert_eq!(None, v.distribution);
            assert_eq!(vec_clauses, v.clauses_head);
            assert_eq!(vec_clauses, v.clauses_body);
            assert!(!state.get_bool(v.bound));
        }
    }
    
    #[test]
    pub fn distribution_creation() {
        let mut g = Graph::default();
        let mut state = StateManager::default();
        let v = g.add_distribution(&vec![0.3, 0.7], &mut state);
        assert_eq!(2, g.variables.len());
        assert_eq!(2, v.len());
        assert_eq!(VariableIndex(0), v[0]);
        assert_eq!(VariableIndex(1), v[1]);
        assert_eq!(0.3, g.get_variable_weight(v[0]).unwrap());
        assert_eq!(0.7, g.get_variable_weight(v[1]).unwrap());
        assert_eq!(DistributionIndex(0), g.get_variable_distribution(v[0]).unwrap());
        assert_eq!(DistributionIndex(0), g.get_variable_distribution(v[1]).unwrap());
        let d = g.distributions[0];
        assert_eq!(VariableIndex(0), d.first);
        assert_eq!(2, d.size);
        let v = g.add_distribution(&vec![0.6, 0.4], &mut state);
        assert_eq!(4, g.variables.len());
        assert_eq!(2, v.len());
        assert_eq!(VariableIndex(2), v[0]);
        assert_eq!(VariableIndex(3), v[1]);
        assert_eq!(0.6, g.get_variable_weight(v[0]).unwrap());
        assert_eq!(0.4, g.get_variable_weight(v[1]).unwrap());
        assert_eq!(DistributionIndex(1), g.get_variable_distribution(v[0]).unwrap());
        assert_eq!(DistributionIndex(1), g.get_variable_distribution(v[1]).unwrap());
        let d = g.distributions[1];
        assert_eq!(VariableIndex(2), d.first);
        assert_eq!(2, d.size);
    }
    
    #[test]
    pub fn clauses_creation() {
        let mut g = Graph::default();
        let mut state = StateManager::default();
        let _ds = (0..3).map(|_| g.add_distribution(&vec![0.4], &mut state)).collect::<Vec<Vec<VariableIndex>>>();
        let p = (0..3).map(VariableIndex).collect::<Vec<VariableIndex>>();
        let d = (0..3).map(|_| g.add_variable(false, None, None, &mut state)).collect::<Vec<VariableIndex>>();
        let c1 = g.add_clause(d[0], vec![p[0], p[1]], &mut state);
        let clause = &g.clauses[c1.0];
        assert_eq!(d[0], clause.head);
        assert_eq!(vec![p[0], p[1]], clause.body);
        assert!(state.get_bool(clause.constrained));
        assert_eq!(2, state.get_usize(clause.number_probabilistic));
        
        let c2 = g.add_clause(d[1], vec![d[0], p[2]], &mut state);
        let clause = &g.clauses[c2.0];
        assert_eq!(d[1], clause.head);
        assert_eq!(vec![d[0], p[2]], clause.body);
        assert!(state.get_bool(clause.constrained));
        assert_eq!(1, state.get_usize(clause.number_probabilistic));
        
        let c3 = g.add_clause(d[0], vec![d[1]], &mut state);
        let clause = &g.clauses[c3.0];
        assert_eq!(d[0], clause.head);
        assert_eq!(vec![d[1]], clause.body);
        assert!(state.get_bool(clause.constrained));
        assert_eq!(0, state.get_usize(clause.number_probabilistic));
    }
}