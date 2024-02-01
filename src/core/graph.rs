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


//! This module contains the main data structure of the solver: the implication graph of a set of Horn clause.
//! It is also responsible for storing the distributions of the input problem.
//! 
//! If C1, ..., Cn are n Horn clauses of the form Ii => hi, then the implication graph of these clauses is
//! a graph G = (V, E) such that
//!     1. There is one node per clause
//!     2. There is an edge e from Ci to Cj if hi is included in Ij
//!     
//! The graph structure contains three main vectors containing, respectively, all the variables, all the clauses and
//! all the distributions of the input formula.
//! Each variable, clause and distribution can then be uniquely identified by an index in these vectors.
//! We provide type-safe abstraction to access these vectors.
//! 
//! Once the graph is constructed, no more variables/clause will be inserted. However, some clauses/variables will be
//! deactivated during the search. And during the component detection/propagation the graph is iterated over multiple times.
//! Hence, we must provide an efficient way of accessing the parents/children of a clause.
//! From an implementation point of view, the nodes in the implication graph use three spare-set to store
//!     1. The references (ClauseIndex) to their parents
//!     2. The references (ClauseIndex) to their children
//!     3. The references (VariableIndex) to the varaibles in their body
//!     
//! Moreover, a consequent number of elements are stored using reversible data structures. This allows to backtrack safely to
//! previous states of the search space.

use std::cmp::Ordering;

use search_trail::*;
use super::literal::*;
use super::variable::*;
use super::clause::*;
use super::distribution::*;
use super::watched_vector::WatchedVector;

use rustc_hash::FxHashMap;

/// Abstraction used as a typesafe way of retrieving a `Variable` in the `Graph` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VariableIndex(pub usize);

/// Abstraction used as a typesafe way of retrieving a `Clause` in the `Graph` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ClauseIndex(pub usize);

/// Abstraction used as a typesafe way of retrieving a `Distribution` in the `Graph` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DistributionIndex(pub usize);

/// Abstraction used as a typesafe way of retrieving a watched vector in the `Graph` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct WatchedVectorIndex(pub usize);

/// Data structure representing the Graph.
#[derive(Debug)]
pub struct Graph {
    /// Vector containing the nodes of the graph
    pub variables: Vec<Variable>,
    /// Vector containing the clauses of the graph
    clauses: Vec<Clause>,
    /// Vector containing the distributions of the graph
    distributions: Vec<Distribution>,
    /// Store for each variables the clauses it watches
    watchers: Vec<Vec<ClauseIndex>>,
    /// Index of the first not fixed variable in the variables vector
    min_var_unassigned: ReversibleUsize,
    /// Index of the last not fixed variable in the variables vector
    max_var_unassigned: ReversibleUsize,
    /// bitwise representation of the state (fixed/not fixed) of the variables
    variables_bit: Vec<ReversibleU64>,
    /// bitwise representation of the state (constrained/unconstrained) of the clauses
    clauses_bit: Vec<ReversibleU64>,
}

impl Graph {
    
    // --- GRAPH CREATION --- //

    /// Creates a new empty implication graph
    pub fn new(state: &mut StateManager, n_var: usize, n_clause: usize) -> Self {
        let variables = (0..n_var).map(|i| Variable::new(i, None, None, state)).collect();
        let watchers = (0..n_var).map(|_| vec![]).collect();
        
        let number_word_variable = (n_var / 64) + if n_var % 64 == 0 { 0 } else { 1 };
        let variables_bit = (0..number_word_variable).map(|_| state.manage_u64(!0)).collect();

        let number_word_clause = (n_clause / 64) + if n_clause % 64 == 0 { 0 } else { 1 };
        let clauses_bit = (0..number_word_clause).map(|_| state.manage_u64(!0)).collect();
        Self {
            variables,
            clauses: vec![],
            watchers,
            distributions: vec![],
            min_var_unassigned: state.manage_usize(0),
            max_var_unassigned: state.manage_usize(n_var),
            variables_bit,
            clauses_bit,
        }
    }

    /// Add a distribution to the graph. In this case, a distribution is a set of probabilistic
    /// variable such that
    ///     - The sum of their weights sum up to 1.0
    ///     - Exatctly one of these variable is true in a model of the input formula 
    ///     - None of the variable in the distribution is part of another distribution
    ///
    /// Each probabilstic variable should be part of one distribution.
    /// This functions adds the variable in the vector of variables. They are in a contiguous part of
    /// the vector.
    /// Moreover, the variables are stored by decreasing probability. The mapping from the old
    /// variables index (the ones used in the encoding) and the new one (in the vector) is
    /// returned.
    pub fn add_distributions(
        &mut self,
        distributions: &Vec<Vec<f64>>,
        state: &mut StateManager,
    ) -> FxHashMap<usize, usize> {
        let mut mapping: FxHashMap<usize, usize> = FxHashMap::default();
        let mut current_start = 0;
        for (d_id, weights) in distributions.iter().enumerate() {
            let distribution = Distribution::new(d_id, VariableIndex(current_start), weights.len(), state);
            let distribution_id = DistributionIndex(self.distributions.len());
            self.distributions.push(distribution);
            let mut weight_with_ids = weights.iter().copied().enumerate().map(|(i, w)| (w,i)).collect::<Vec<(f64, usize)>>();
            weight_with_ids.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
            // j is the new index while i is the initial index
            // So the variable will be store at current_start + j instead of current_start + i (+ 1
            // for the mapping because in the input file indexes start at 1)
            for (j, (w, i)) in weight_with_ids.iter().copied().enumerate() {
                let new_index = current_start + j;
                let initial_index = current_start + i;
                self.variables[new_index].set_distribution(distribution_id);
                self.variables[new_index].set_weight(w);
                mapping.insert(initial_index + 1, new_index + 1);
            }
            current_start += weights.len();
        }
        mapping
    }
    
    pub fn add_clause(
        &mut self,
        mut literals: Vec<Literal>,
        head: Option<Literal>,
        state: &mut StateManager,
        is_learned: bool,
    ) -> ClauseIndex {

        let cid = ClauseIndex(self.clauses.len());
        // We sort the literals by probabilistic/non-probabilistic
        let number_probabilistic = literals.iter().copied().filter(|l| self[l.to_variable()].is_probabilitic()).count();
        let number_deterministic = literals.len() - number_probabilistic;
        literals.sort_by(|a, b| {
            let a_var = a.to_variable();
            let b_var = b.to_variable();
            if self[a_var].is_probabilitic() && !self[b_var].is_probabilitic() {
                Ordering::Greater
            } else if !self[a_var].is_probabilitic() && self[b_var].is_probabilitic() {
                Ordering::Less
            } else {
                if self[a_var].decision_level() < self[b_var].decision_level() {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
        });
        
        let literal_vector = WatchedVector::new(literals, number_deterministic, state);
        
        // The first two deterministic variables in the clause watch the clause
        for i in 0..number_deterministic.min(2) {
            let variable = literal_vector[i].to_variable();
            self.watchers[variable.0].push(cid);
        }
        
        for i in 0..number_probabilistic.min(2) {
            let variable = literal_vector[number_deterministic + i].to_variable();
            self.watchers[variable.0].push(cid);
        }
        
        let mut clause = Clause::new(cid.0, literal_vector, head, is_learned, state);
        // If the clause is not learned, we need to link it to the other clauses for FT-reachable propagation.
        for literal in clause.iter().collect::<Vec<Literal>>() {
            let variable = literal.to_variable();
            if literal.is_positive() {
                self[variable].add_clause_positive_occurence(cid);
                if !is_learned {
                    for child in self[variable].iter_clauses_negative_occurence().collect::<Vec<ClauseIndex>>() {
                        clause.add_child(child, state);
                        self[child].add_parent(cid, state);
                    }
                }
            } else {
                self[variable].add_clause_negative_occurence(cid);
                if !is_learned {
                    for parent in self[variable].iter_clauses_positive_occurence().collect::<Vec<ClauseIndex>>() {
                        clause.add_parent(parent, state);
                        self[parent].add_child(cid, state);
                    }
                }
            }
            if let Some(distribution) = self[literal.to_variable()].distribution() {
                if !is_learned {
                    self[distribution].increment_clause();
                }
            }
        }
        self.clauses.push(clause);
        cid
    }
    
    // --- GRAPH MODIFICATIONS --- //
    
    /// Sets a variable to true or false.
    ///     - If true, Removes the variable from the body of the constrained clauses
    ///     - If false, and probabilistic, increase the counter of false variable in the distribution
    /// If the variable is the min or max variable not fixed, update the boundaries accordingly.
    pub fn set_variable(&mut self, variable: VariableIndex, value: bool, level: isize, reason: Option<Reason>, state: &mut StateManager) {
        self[variable].set_value(value, state);
        self[variable].set_decision_level(level);
        self[variable].set_reason(reason, state);
        
        // Updating the bitwise representation of the variables state
        let bit_vec_idx = variable.0 / 64;
        let bit_idx = variable.0 % 64;
        let cur_word = state.get_u64(self.variables_bit[bit_vec_idx]);
        state.set_u64(self.variables_bit[bit_vec_idx], cur_word & !(1 << bit_idx));

        // If probabilistic and false, update the counter
        if !value && self[variable].is_probabilitic() {
            let distribution = self[variable].distribution().unwrap();
            self[distribution].increment_number_false(state);
            self[distribution].remove_probability_mass(self[variable].weight().unwrap(), state);
        }

        //  Update the boundaries of min/max variable not fixed if necessary
        if variable.0 == state.get_usize(self.min_var_unassigned) {
            let mut cur = variable;
            let end = state.get_usize(self.max_var_unassigned);
            while cur.0 <= end && self[cur].is_fixed(state) {
                cur += 1;
            }
            state.set_usize(self.min_var_unassigned, cur.0);
        }
        if variable.0 == state.get_usize(self.max_var_unassigned) {
            let mut cur = variable;
            let end = state.get_usize(self.min_var_unassigned);
            while cur.0 >= end && self[cur].is_fixed(state) {
                cur -= 1;
            }
            state.set_usize(self.max_var_unassigned, cur.0);
        }
    }

    /// Returns the number of clauses watched by the variable
    pub fn number_watchers(&self, variable: VariableIndex) -> usize {
        self.watchers[variable.0].len()
    }

    /// Returns the clause watched by the variable at id watcher_id
    pub fn get_clause_watched(&self, variable: VariableIndex, watcher_id: usize) -> ClauseIndex {
        self.watchers[variable.0][watcher_id]
    }
    
    pub fn remove_watcher(&mut self, variable: VariableIndex, watcher_id: usize) {
        self.watchers[variable.0].swap_remove(watcher_id);
    }
    
    pub fn add_watcher(&mut self, variable: VariableIndex, clause: ClauseIndex) {
        self.watchers[variable.0].push(clause);
    }

    /// Decrements the number of constrained clauses a distribution is in
    pub fn decrement_distribution_constrained_clause_counter(&self, distribution: DistributionIndex, state: &mut StateManager) -> usize {
        state.decrement_usize(self.distributions[distribution.0].number_clause_unconstrained)
    }
    
    // --- QUERIES --- //
    
    /// Set a clause as unconstrained
    pub fn set_clause_unconstrained(&self, clause: ClauseIndex, state: &mut StateManager) {
        self[clause].set_unconstrained(state);
        
        // Update the bitwise representation
        if !self[clause].is_learned() {
            let id = self.clauses_bit[clause.0 / 64];
            let w = state.get_u64(id);
            state.set_u64(id, w & !(1 << (clause.0 % 64)));
        }
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
    
    /// Returns the number of distribution in the graph
    pub fn number_distributions(&self) -> usize {
        self.distributions.len()
    }
    
    // --- ITERATORS --- //
    
    /// Returns an iterator on all (constrained and unconstrained) the clauses of the graph
    pub fn clauses_iter(&self) -> impl Iterator<Item = ClauseIndex> {
        (0..self.clauses.len()).map(ClauseIndex)
    }

    /// Returns an iterator on the distributions of the problem
    pub fn distributions_iter(&self) -> impl Iterator<Item = DistributionIndex> {
        (0..self.distributions.len()).map(DistributionIndex)
    }
    
    pub fn variables_iter(&self) -> impl Iterator<Item = VariableIndex> {
        (0..self.variables.len()).map(VariableIndex)
    }
}

// --- Indexing the graph with the various indexes --- //

impl std::ops::Index<VariableIndex> for Graph {
    type Output = Variable;

    fn index(&self, index: VariableIndex) -> &Self::Output {
        &self.variables[index.0]
    }
}

impl std::ops::IndexMut<VariableIndex> for Graph {
    fn index_mut(&mut self, index: VariableIndex) -> &mut Self::Output {
        &mut self.variables[index.0]
    }
}

impl std::ops::Index<ClauseIndex> for Graph {
    type Output = Clause;

    fn index(&self, index: ClauseIndex) -> &Self::Output {
        &self.clauses[index.0]
    }
}

impl std::ops::IndexMut<ClauseIndex> for Graph {
    fn index_mut(&mut self, index: ClauseIndex) -> &mut Self::Output {
        &mut self.clauses[index.0]
    }
}


impl std::ops::Index<DistributionIndex> for Graph {
    type Output = Distribution;

    fn index(&self, index: DistributionIndex) -> &Self::Output {
        &self.distributions[index.0]
    }
}

impl std::ops::IndexMut<DistributionIndex> for Graph {
    fn index_mut(&mut self, index: DistributionIndex) -> &mut Self::Output {
        &mut self.distributions[index.0]
    }
}

// --- Operator on the indexes for the vectors --- //

impl std::ops::Add<usize> for VariableIndex {
    type Output = VariableIndex;

    fn add(self, rhs: usize) -> Self::Output {
        VariableIndex(self.0 + rhs)   
    }
}

impl std::ops::AddAssign<usize> for VariableIndex {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs;
    }
}

impl std::ops::Sub<usize> for VariableIndex {
    type Output = VariableIndex;

    fn sub(self, rhs: usize) -> Self::Output {
        VariableIndex(self.0 - rhs)
    }
}

impl std::ops::SubAssign<usize> for VariableIndex {
    fn sub_assign(&mut self, rhs: usize) {
        self.0 -= rhs;
    }
}

/*
#[cfg(test)]
mod test_graph_creation {
    
    use crate::core::graph::*;
    use search_trail::StateManager;

    #[test]
    pub fn variables_creation() {
        let mut state = StateManager::default();
        let mut g = Graph::new(&mut state);
        let x = (0..3).map(|x| g.add_variable(true, Some(1.0 / (x + 1) as f64), Some(DistributionIndex(x)), &mut state)).collect::<Vec<VariableIndex>>();
        for i in 0..3 {
            assert_eq!(VariableIndex(i), x[i]);
            let v = &g.variables[x[i].0];
            let vec_clause: Vec<ClauseIndex> = vec![];
            assert!(v.probabilistic);
            assert_eq!(Some(1.0 / (i + 1) as f64), v.weight());
            assert_eq!(Some(DistributionIndex(i)), v.distribution());
            assert_eq!(vec_clause, v.iter_clauses_negative_occurence().collect::<Vec<ClauseIndex>>());
            assert_eq!(vec_clause, v.iter_clause_negative_occurence().collect::<Vec<ClauseIndex>>());
            assert!(v.value(&state).is_none());
        }

        let x = (0..3).map(|_| g.add_variable(false, None, None, &mut state)).collect::<Vec<VariableIndex>>();
        for i in 0..3 {
            assert_eq!(VariableIndex(i+3), x[i]);
            let v = &g.variables[x[i].0];
            let vec_clauses: Vec<ClauseIndex> = vec![];
            assert!(!v.probabilistic);
            assert_eq!(None, v.weight);
            assert_eq!(None, v.distribution);
            assert_eq!(vec_clauses, v.clauses_positive);
            assert_eq!(vec_clauses, v.clauses_negative);
            assert!(state.get_option_bool(v.value).is_none());
        }
    }
    
    #[test]
    pub fn distribution_creation() {
        let mut state = StateManager::default();
        let mut g = Graph::new(&mut state);
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
        let mut state = StateManager::default();
        let mut g = Graph::new(&mut state);
        let _ds = (0..3).map(|_| g.add_distribution(&vec![0.4], &mut state)).collect::<Vec<Vec<VariableIndex>>>();
        let p = (0..3).map(VariableIndex).collect::<Vec<VariableIndex>>();
        let d = (0..3).map(|_| g.add_variable(false, None, None, &mut state)).collect::<Vec<VariableIndex>>();
        let c1 = g.add_clause(d[0], vec![p[0], p[1]], &mut state);
        let clause = &g.clauses[c1.0];
        assert_eq!(d[0], clause.head);
        assert_eq!(vec![p[0], p[1]], g.clause_body_iter(c1).collect::<Vec<VariableIndex>>());
        assert!(state.get_bool(clause.constrained));
        
        let c2 = g.add_clause(d[1], vec![d[0], p[2]], &mut state);
        let clause = &g.clauses[c2.0];
        assert_eq!(d[1], clause.head);
        assert_eq!(vec![p[2], d[0]], g.clause_body_iter(c2).collect::<Vec<VariableIndex>>());
        assert!(state.get_bool(clause.constrained));
        
        let c3 = g.add_clause(d[0], vec![d[1]], &mut state);
        let clause = &g.clauses[c3.0];
        assert_eq!(d[0], clause.head);
        assert_eq!(vec![d[1]], g.clause_body_iter(c3).collect::<Vec<VariableIndex>>());
        assert!(state.get_bool(clause.constrained));
    }
}

#[cfg(test)]
mod graph_update {
    
    use crate::core::graph::*;
    use search_trail::StateManager;
    
    fn get_graph(state: &mut StateManager) -> Graph {
        // Structure of the graph:
        //
        // c0 --> c2 --> c3
        //        ^       |
        //        |       v
        // c1 ----------> c4
        let mut g = Graph::new(state);
        let nodes: Vec<VariableIndex> = (0..10).map(|_| g.add_variable(false, None, None, state)).collect();
        let _c0 = g.add_clause(nodes[2], vec![nodes[0], nodes[1]], state);
        let _c1 = g.add_clause(nodes[5], vec![nodes[3], nodes[4]], state);
        let _c2 = g.add_clause(nodes[6], vec![nodes[2], nodes[5]], state);
        let _c3 = g.add_clause(nodes[8], vec![nodes[6], nodes[7]], state);
        let _c4 = g.add_clause(nodes[9], vec![nodes[5], nodes[8]], state);
        g
    }
    
    fn check_parents(graph: &Graph, clause: ClauseIndex, mut expected_parents: Vec<usize>, state: &StateManager) {
        let mut actual = graph.parents_clause_iter(clause, state).map(|c| c.0).collect::<Vec<usize>>();
        actual.sort();
        expected_parents.sort();
        assert_eq!(expected_parents, actual);
    }
    
    fn check_children(graph: &Graph, clause: ClauseIndex, mut expected_children: Vec<usize>, state: &StateManager) {
        let mut actual = graph.children_clause_iter(clause, state).map(|c| c.0).collect::<Vec<usize>>();
        actual.sort();
        expected_children.sort();
        assert_eq!(expected_children, actual);
    }
    
    #[test]
    fn remove_clause() {
        let mut state = StateManager::default();
        let mut g = get_graph(&mut state);
        check_parents(&g, ClauseIndex(0), vec![], &state);
        check_parents(&g, ClauseIndex(1), vec![], &state);
        check_parents(&g, ClauseIndex(2), vec![0, 1], &state);
        check_parents(&g, ClauseIndex(3), vec![2], &state);
        check_parents(&g, ClauseIndex(4), vec![1, 3], &state);
        
        check_children(&g, ClauseIndex(0), vec![2], &state);
        check_children(&g, ClauseIndex(1), vec![2, 4], &state);
        check_children(&g, ClauseIndex(2), vec![3], &state);
        check_children(&g, ClauseIndex(3), vec![4], &state);
        check_children(&g, ClauseIndex(4), vec![], &state);
        
        g.remove_clause_from_children(ClauseIndex(2), &mut state);
        g.remove_clause_from_parent(ClauseIndex(2), &mut state);

        check_parents(&g, ClauseIndex(0), vec![], &state);
        check_parents(&g, ClauseIndex(1), vec![], &state);
        check_parents(&g, ClauseIndex(2), vec![0, 1], &state);
        check_parents(&g, ClauseIndex(3), vec![], &state);
        check_parents(&g, ClauseIndex(4), vec![1, 3], &state);

        check_children(&g, ClauseIndex(0), vec![], &state);
        check_children(&g, ClauseIndex(1), vec![4], &state);
        check_children(&g, ClauseIndex(2), vec![3], &state);
        check_children(&g, ClauseIndex(3), vec![4], &state);
        check_children(&g, ClauseIndex(4), vec![], &state);

        g.remove_clause_from_children(ClauseIndex(1), &mut state);
        g.remove_clause_from_parent(ClauseIndex(1), &mut state);

        check_parents(&g, ClauseIndex(0), vec![], &state);
        check_parents(&g, ClauseIndex(3), vec![], &state);
        check_parents(&g, ClauseIndex(4), vec![3], &state);

        check_children(&g, ClauseIndex(0), vec![], &state);
        check_children(&g, ClauseIndex(3), vec![4], &state);
        check_children(&g, ClauseIndex(4), vec![], &state);

    }

}

#[cfg(test)]
mod get_bit_representation {

    use crate::core::graph::*;
    use search_trail::StateManager;
    
    fn check_bit_array(expected: &Vec<u64>, actual: &Vec<ReversibleU64>, state: &StateManager) {
        assert_eq!(expected.len(), actual.len());
        for i in 0..expected.len() {
            assert_eq!(expected[i], state.get_u64(actual[i]));
        }
    }
    
    #[test]
    fn initial_representation() {
        let mut state = StateManager::default();
        let mut graph = Graph::new(&mut state);
        check_bit_array(&vec![], &graph.variables_bit, &state);
        for i in 0..300 {
            graph.add_variable(false, None, None, &mut state);
            if i % 64 == 0 {
                check_bit_array(&vec![!0; (i / 64)+1], &graph.variables_bit, &state);
            }
        }
        check_bit_array(&vec![!0; 5], &graph.variables_bit, &state);

        check_bit_array(&vec![], &graph.clauses_bit, &state);
        for i in 0..300 {
            graph.add_clause(VariableIndex(i), vec![VariableIndex(i)], &mut state);
            if i % 64 == 0 {
                check_bit_array(&vec![!0; (i / 64)+1], &graph.clauses_bit, &state);
            }
        }
        check_bit_array(&vec![!0; 5], &graph.clauses_bit, &state);
    }
    
    fn deactivate_bit(bit_array: &mut Vec<u64>, word_index: usize, bit_index: usize) {
        bit_array[word_index] &= !(1 << bit_index);
    }
    
    #[test]
    fn update_representation() {
        let mut state = StateManager::default();
        let mut graph = Graph::new(&mut state);
        for _ in 0..300 {
            graph.add_variable(false, None, None, &mut state);
        }
        for i in 0..300 {
            graph.add_clause(VariableIndex(i), vec![VariableIndex(i)], &mut state);
        }
        let mut expected = vec![!0; 5];

        graph.set_variable(VariableIndex(0), false, 0, None, &mut state);
        graph.set_clause_unconstrained(ClauseIndex(0), &mut state);
        deactivate_bit(&mut expected, 0, 0);
        check_bit_array(&expected, &graph.variables_bit, &state);
        check_bit_array(&expected, &graph.clauses_bit, &state);

        graph.set_variable(VariableIndex(130), false, 0, None, &mut state);
        graph.set_clause_unconstrained(ClauseIndex(130), &mut state);
        deactivate_bit(&mut expected, 2, 2);
        check_bit_array(&expected, &graph.variables_bit, &state);
        check_bit_array(&expected, &graph.clauses_bit, &state);

        graph.set_variable(VariableIndex(131), false, 0, None, &mut state);
        graph.set_clause_unconstrained(ClauseIndex(131), &mut state);
        deactivate_bit(&mut expected, 2, 3);
        check_bit_array(&expected, &graph.variables_bit, &state);
        check_bit_array(&expected, &graph.clauses_bit, &state);

        graph.set_variable(VariableIndex(299), false, 0, None, &mut state);
        graph.set_clause_unconstrained(ClauseIndex(299), &mut state);
        deactivate_bit(&mut expected, 4, 43);
        check_bit_array(&expected, &graph.variables_bit, &state);
        check_bit_array(&expected, &graph.clauses_bit, &state);
    }
}
*/
