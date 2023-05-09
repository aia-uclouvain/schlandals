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

use crate::core::components::{ComponentIndex, ComponentExtractor};
use search_trail::*;
use crate::solver::sequential::CacheEntry;
use crate::core::sparse_set::SparseSet;

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

/// Data structure that actually holds the data of a  variable of the input problem
#[derive(Debug, Clone)]
pub struct Variable {
    /// True if and only if the variable is probabilistic
    pub probabilistic: bool,
    /// If `probabilistic` is `true`, then this is the weight associated to the variable. Otherwise,
    /// this is None.
    pub weight: Option<f64>,
    /// If `probabilistic` is `true`, this is the index of the distribution containing this node. Otherwise,
    /// this is None.
    pub distribution: Option<DistributionIndex>,
    /// The clauses (I => h) in which the variable appears as the head (h)
    pub clauses_head: Vec<ClauseIndex>,
    /// The clauses (I => h) in which the variable appears in the body (I)
    pub clauses_body: Vec<ClauseIndex>,
    /// The value assigned to the variable
    value: bool,
    /// True if and only if the variable is fixed
    fixed: ReversibleBool,
    /// Number of constrained clause containing the variable
    number_clause_constrained: ReversibleUsize,
}

/// This structure represent a clause in the graph. A clause is of the form
///     a && b && c => d
/// The body of the clause (a, b, c) are stored in  a reversible sparse-set.
/// This allows to efficiently iterate over the body without considering fixed variables.
/// The same is done for the parents and children in the implication graph.
/// A clause can be in two states:
///     1) Constrained: we still need to consider it during the solving of the problem
///     2) Unconstrained: it always evaluates to T, independetly of the choices for the other variables.
/// 
/// If a clause is unconstrained, it is not considered during the component detection, branching decision, etc.
/// The body, parents and children are stored in reversible sparse-set for efficient update/iteration.
#[derive(Debug)]
pub struct Clause {
    /// Head of the clause
    pub head: VariableIndex,
    /// Body of the clause, cunjunction of all the variable in the body
    pub body: SparseSet<VariableIndex>,
    /// True if and only if the clause is constrained
    pub constrained: ReversibleBool,
    /// Numbers of probabilistic variable active in the clause body
    pub number_probabilistic: ReversibleUsize,
    /// Numbers of deterministic variable active in the clause body
    pub number_deterministic: ReversibleUsize,
    /// Vector that stores the children of the clause in the implication graph
    pub children: SparseSet<ClauseIndex>,
    /// Vector that stores the parents of the clause in the implication graph
    pub parents: SparseSet<ClauseIndex>,
}

/// Represents a set of variable in a same distribution. This assume that the variable of a distribution
/// are inserted in the graph one after the other (i.e. that their `VariableIndex` are consecutive).
/// Since no variable should be removed from the graph once constructed, this should not be a problem.
/// Thus a distribution is identified by the first `VariableIndex` and the number of variable in it.
/// 
/// We also store the number of clauses in which the distribution appears. This is usefull to compute
/// the unconstrained probability during the propagation.
/// In the same manner, the number of variable assigned to false in the distribution is kept, allowing us
/// to efficiently detect that only one variable remains in a distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Distribution {
    /// First variable in the distribution
    pub first: VariableIndex,
    /// Number of variable in the distribution
    pub size: usize,
    /// Number of constrained clauses in which the distribution appears
    pub number_clause_constrained: ReversibleUsize,
    /// Number of variables assigned to F in the distribution
    pub number_false: ReversibleUsize,
}

/// Data structure representing the Graph.
#[derive(Debug)]
pub struct Graph {
    /// Vector containing the nodes of the graph
    variables: Vec<Variable>,
    /// Vector containing the clauses of the graph
    clauses: Vec<Clause>,
    /// Vector containing the distributions of the graph
    distributions: Vec<Distribution>,
    /// Number of probabilistic nodes in the graph
    number_probabilistic: usize,
    /// Index of the first not fixed variable in the variables vector
    min_var_unassigned: ReversibleUsize,
    /// Index of the last not fixed variable in the variables vector
    max_var_unassigned: ReversibleUsize,
    /// bitwise representation of the state (fixed/not fixed) of the variables
    variables_bit: Vec<ReversibleU64>,
    /// bitwise representation of the state (constrained/unconstrained) of the clauses
    clauses_bit: Vec<ReversibleU64>,
    /// Random bitstring for the variables (used in hash computation)
    variables_random: Vec<u64>,
    /// Random bitstring for the clauses (used in hash computation)
    clauses_random: Vec<u64>,
}

impl Graph {
    
    // --- GRAPH CREATION --- //

    /// Creates a new empty implication graph
    pub fn new(state: &mut StateManager) -> Self {
        Self {
            variables: vec![],
            clauses: vec![],
            distributions: vec![],
            number_probabilistic: 0,
            min_var_unassigned: state.manage_usize(0),
            max_var_unassigned: state.manage_usize(0),
            variables_bit: vec![],
            clauses_bit: vec![],
            variables_random: vec![],
            clauses_random: vec![],
        }
    }

    /// Add a distribution to the graph. In this case, a distribution is a set of probabilistic
    /// variable such that
    ///     - The sum of their weights sum up to 1.0
    ///     - Exatctly one of these variable is true in a solution
    ///     - None of the variable in the distribution is part of another distribution
    ///
    /// Each probabilstic variable should be part of one distribution.
    /// This functions adds the variable in the vector of variables. They are in a contiguous part of
    /// the vector.
    /// Returns the index of the variables in the distribution
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
            number_clause_constrained: state.manage_usize(0),
            number_false: state.manage_usize(0),
        });
        variables 
    }
    
    /// Adds a variable to the graph and returns its index
    pub fn add_variable(
        &mut self,
        probabilistic: bool,
        weight: Option<f64>,
        distribution: Option<DistributionIndex>,
        state: &mut StateManager,
    ) -> VariableIndex {
        // Random bitstring for the variable, used in the hash computation
        self.variables_random.push(rand::random());
        // Since a variable is added, increase the max unfixed variable
        state.set_usize(self.max_var_unassigned, self.variables.len());
        let id = self.variables.len();
        // If needed, add a word for the bitwise representation
        if id / 64 >= self.variables_bit.len() {
            self.variables_bit.push(state.manage_u64(!0));
        }
        self.variables.push(Variable {
            probabilistic,
            weight,
            distribution,
            clauses_head: vec![],
            clauses_body: vec![],
            value: false,
            fixed: state.manage_bool(false),
            number_clause_constrained: state.manage_usize(0),
        });
        VariableIndex(id)
    }

    /// Add a clause to the graph. A clause is a expression of the form
    ///     v1 && v2 && ... && nn => head
    ///
    /// where head, v1, ..., vn are variable of the graph. ´head` is the head of the clause and `body`
    /// = vec![v1, ..., vn].
    pub fn add_clause(
        &mut self,
        head: VariableIndex,
        body: Vec<VariableIndex>,
        state: &mut StateManager,
    ) -> ClauseIndex {
        // Random bitstring for the clause, used in hash computation
        self.clauses_random.push(rand::random());
        // If needed, increase number of words for the bitwise representation of the problem
        if self.clauses.len() / 64 >= self.clauses_bit.len() {
            self.clauses_bit.push(state.manage_u64(!0));
        }
        let cid = ClauseIndex(self.clauses.len());
        let number_probabilistic = body.iter().copied().filter(|v| self.is_variable_probabilistic(*v)).count();
        let number_deterministic = body.len() - number_probabilistic;
        self.clauses.push(Clause {
            head,
            body: SparseSet::<VariableIndex>::new(state),
            constrained: state.manage_bool(true),
            number_probabilistic: state.manage_usize(number_probabilistic),
            number_deterministic: state.manage_usize(number_deterministic),
            children: SparseSet::<ClauseIndex>::new(state),
            parents: SparseSet::<ClauseIndex>::new(state),
        });
        // This part of the code sets up all the necessary counters, references for the propagation algorithm
        state.increment_usize(self.variables[head.0].number_clause_constrained);
        for n in body {
            if self.is_variable_probabilistic(n) {
                // We increment the number of clause containing the distribution. Note that it is not a problem
                // if two variables from the same distribution are in the body. When they are removed, the count
                // is decremented multiple times
                let distribution = self.variables[n.0].distribution.unwrap();
                state.increment_usize(self.distributions[distribution.0].number_clause_constrained);
            }
            state.increment_usize(self.variables[n.0].number_clause_constrained);
            // Since n is in the body of the current clause, an edge must be added from each clause in which it is
            // the head
            for clause in self.variables[n.0].clauses_head.clone() {
                self.add_edge(clause, cid, state);
            }
            self.variables[n.0].clauses_body.push(cid);
            self.clauses[cid.0].body.add(n, state);
        }
        // For each clause in which the head is in the body, add an edge from the current clause to that clause
        for clause in self.variables[head.0].clauses_body.clone() {
            self.add_edge(cid, clause, state);
        }
        self.variables[head.0].clauses_head.push(cid);
        // Set numbers of probabilistic/deterministic for reachability computations
        state.set_usize(self.clauses[cid.0].number_probabilistic, number_probabilistic);
        state.set_usize(self.clauses[cid.0].number_deterministic, number_deterministic);
        cid
    }

    /// Add an edge between the clause identified by `src` to the clause identified by `dst`. This
    /// method returns the index of the edge.
    fn add_edge(
        &mut self,
        src: ClauseIndex,
        dst: ClauseIndex,
        state: &mut StateManager,
    ) {
        self.clauses[src.0].children.add(dst, state);
        self.clauses[dst.0].parents.add(src, state);
    }
    
    // --- GRAPH MODIFICATIONS --- //
    
    /// Removes a child from the sparse-set of children of the clause
    fn remove_child(&mut self, clause: ClauseIndex, child: ClauseIndex, state: &mut StateManager) {
        self.clauses[clause.0].children.remove(child, state);
    }

    /// Removes a parent from the sparse-set of parents of the clause
    fn remove_parent(&mut self, clause: ClauseIndex, parent: ClauseIndex, state: &mut StateManager) {
        self.clauses[clause.0].parents.remove(parent, state);
    }
    
    /// Removes a variable from the sparse-set of the body of the clause
    fn remove_variable_from_body(&mut self, variable: VariableIndex, clause: ClauseIndex, state: &mut StateManager) {
        self.clauses[clause.0].body.remove(variable, state);
    }
    
    /// Sets a variable to true or false.
    ///     - If true, Removes the variable from the body of the constrained clauses
    ///     - If false, and probabilistic, increase the counter of false variable in the distribution
    /// If the variable is the min or max variable not fixed, update the boundaries accordingly.
    pub fn set_variable(&mut self, variable: VariableIndex, value: bool, state: &mut StateManager) {
        state.set_bool(self.variables[variable.0].fixed, true);
        self.variables[variable.0].value = value;
        
        // Updating the bitwise representation of the variables state
        let bit_vec_idx = variable.0 / 64;
        let bit_idx = variable.0 % 64;
        let cur_word = state.get_u64(self.variables_bit[bit_vec_idx]);
        state.set_u64(self.variables_bit[bit_vec_idx], cur_word & !(1 << bit_idx));
        
        // Removes from the body of the clauses it is in
        if value {
            for i in 0..self.variables[variable.0].clauses_body.len() {
                let clause = self.variables[variable.0].clauses_body[i];
                if self.is_clause_constrained(clause, state) {
                    self.remove_variable_from_body(variable, clause, state);
                }
            }
        }
        
        // If probabilistic and false, update the counter
        if !value && self.is_variable_probabilistic(variable) {
            let d = self.get_variable_distribution(variable).unwrap();
            state.increment_usize(self.distributions[d.0].number_false);
        }

        //  Update the boundaries of min/max variable not fixed if necessary
        if variable.0 == state.get_usize(self.min_var_unassigned) {
            let mut cur = variable.0;
            let end = state.get_usize(self.max_var_unassigned);
            while cur <= end && state.get_bool(self.variables[cur].fixed) {
                cur += 1;
            }
            state.set_usize(self.min_var_unassigned, cur);
        }
        if variable.0 == state.get_usize(self.max_var_unassigned) {
            let mut cur = variable.0;
            let end = state.get_usize(self.min_var_unassigned);
            while cur >= end && state.get_bool(self.variables[cur].fixed) {
                cur -= 1;
            }
            state.set_usize(self.max_var_unassigned, cur);
        }
    }

    /// Decrements the number of constrained clauses a distribution is in
    pub fn decrement_distribution_constrained_clause_counter(&self, distribution: DistributionIndex, state: &mut StateManager) -> usize {
        state.decrement_usize(self.distributions[distribution.0].number_clause_constrained)
    }
    
    /// Decrements the number of unassigned probabilistic variables in the body of a clause and returns the remaining
    /// number of unassigned variables in the body
    pub fn clause_decrement_number_probabilistic(&self, clause: ClauseIndex, state: &mut StateManager) -> usize {
        let remaining = state.get_usize(self.clauses[clause.0].number_deterministic);
        remaining + state.decrement_usize(self.clauses[clause.0].number_probabilistic)
    }

    /// Decrements the number of unassigned deterministic variables in the body of a clause and returns the remaining
    /// number of unassigned variables in the body
    pub fn clause_decrement_number_deterministic(&self, clause: ClauseIndex, state: &mut StateManager) -> usize {
        let remaining = state.get_usize(self.clauses[clause.0].number_probabilistic);
        remaining + state.decrement_usize(self.clauses[clause.0].number_deterministic)
    }
    
    // --- QUERIES --- //
    
    /// Gets the cache key for the current subproblem. It is composed as follows
    ///     1. The hash of the subproblem
    ///     2. The u64 representing the minimum unassigned variable (vmin)
    ///     3. The u64 representing the maximum unassigned variable (vmax)
    ///     4. The bitwise representation of the status (fixed/not fixed), from vmin to vmax
    ///     5. The bitwise represnetation of the status (constrained/unconstrained) of the clauses
    /// 
    /// In addition to that, the hash (xor of the random bitstring for the variables/clauses) is returned as a cache entry struct.
    /// The idea is that using a xor for the hash is not perfect, two subproblems might have the same hash but be different. To avoid
    /// returning a wrong result, the bitwise representation is given with the hash, starting from elements that are likely to be different
    /// in different subproblem (min-max unfixed variable and variable status).
    pub fn get_bit_representation(&mut self, state: &StateManager, component: ComponentIndex, extractor: &ComponentExtractor) -> CacheEntry{
        let vmin = state.get_usize(self.min_var_unassigned);
        let vmax = state.get_usize(self.max_var_unassigned);
        let chash = extractor.get_comp_hash(component);
        let mut v: Vec<u64> = vec![vmin as u64, vmax as u64];
        for u in self.variables_bit[(vmin/64)..((vmax/64+1))].iter() {
            v.push(state.get_u64(*u));
        }
        let cls = extractor.component_iter(component).collect::<Vec<ClauseIndex>>();
        for clause in cls {
            v.push(state.get_u64(self.clauses_bit[clause.0 / 64]));

        }
        CacheEntry::new(chash, v)
    }

    /// Returns true if the clause is bound. This happens if the size of the clause is 0
    pub fn is_clause_constrained(&self, clause: ClauseIndex, state: &StateManager) -> bool {
        state.get_bool(self.clauses[clause.0].constrained)
    }
    
    /// Returns true if the variable is bound
    pub fn is_variable_fixed(&self, variable: VariableIndex, state: &StateManager) -> bool {
        state.get_bool(self.variables[variable.0].fixed)
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
        self.clauses[clause.0].body.len(state)
    }
    
    /// Removes the current clause from the child sparse-set of all its parents
    pub fn remove_clause_from_parent(&mut self, clause: ClauseIndex, state: &mut StateManager) {
        for parent in self.parents_clause_iter(clause, state).collect::<Vec<ClauseIndex>>() {
            self.remove_child(parent, clause, state);
        }
    }
    
    /// removes the current clause from the parent sparse-set of all its children
    pub fn remove_clause_from_children(&mut self, clause: ClauseIndex, state: &mut StateManager) {
        for child in self.children_clause_iter(clause, state).collect::<Vec<ClauseIndex>>() {
            self.remove_parent(child, clause, state);
        }
    }
    
    /// Set a clause as unconstrained
    pub fn set_clause_unconstrained(&self, clause: ClauseIndex, state: &mut StateManager) {
        state.set_bool(self.clauses[clause.0].constrained, false);
        
        // Update the bitwise representation
        let id = self.clauses_bit[clause.0 / 64];
        let w = state.get_u64(id);
        state.set_u64(id, w & !(1 << (clause.0 % 64)));
    }
    
    /// Returns true if there are only one variable not fixed in the distribution, and every other is F
    pub fn distribution_one_left(&self, distribution: DistributionIndex, state: &StateManager) -> bool {
        self.distributions[distribution.0].size - state.get_usize(self.distributions[distribution.0].number_false) == 1
    }
    
    /// Returns the number of false variable in the distribution
    pub fn distribution_number_false(&self, distribution: DistributionIndex, state: &StateManager) -> usize {
        state.get_usize(self.distributions[distribution.0].number_false)
    }
    
    // --- GETTERS --- //

    /// Returns the random bitstring of the clause
    pub fn get_clause_random(&self, clause: ClauseIndex) -> u64 {
        self.clauses_random[clause.0]
    }
    
    /// Returns the random bitstring of the variable
    pub fn get_variable_random(&self, variable: VariableIndex) -> u64 {
        self.variables_random[variable.0]
    }
    
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

    /// Returns the value of the vairable, or None if the variable is not fixed
    pub fn get_variable_value(&self, variable: VariableIndex, state: &StateManager) -> Option<bool> {
        if !self.is_variable_fixed(variable, state) {
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
    
    /// Returns the distribution of the first probabilistic (not fixed) variables in the body, or None if the distribution
    /// does not have any not fixed probabilistic variables in its body.
    pub fn get_clause_active_distribution(&self, clause: ClauseIndex, state: &StateManager) -> Option<DistributionIndex> {
        self.clause_body_iter(clause, state).filter(|v| self.is_variable_probabilistic(*v) && !self.is_variable_fixed(*v, state)).map(|v| self.get_variable_distribution(v).unwrap()).next()
    }
    
    /// Returns the number of constrained parents of the clause
    pub fn get_clause_number_parents(&self, clause: ClauseIndex, state: &StateManager) -> usize {
        self.clauses[clause.0].parents.len(state)
    }
    
    /// Returns the number of unconstrained parents of the clause
    pub fn get_clause_removed_parents(&self, clause: ClauseIndex, state: &StateManager) -> usize {
        self.clauses[clause.0].parents.capacity() - self.clauses[clause.0].parents.len(state)
    }

    /// Returns the number of constrained children of the clause
    pub fn get_clause_number_children(&self, clause: ClauseIndex, state: &StateManager) -> usize {
        self.clauses[clause.0].children.len(state)
    }
    
    /// Returns the number of unconstrained children of the clause
    pub fn get_clause_removed_children(&self, clause: ClauseIndex, state: &StateManager) -> usize {
        self.clauses[clause.0].children.capacity() - self.clauses[clause.0].children.len(state)
    }

    // --- ITERATORS --- //
    
    /// Returns an iterator on the constrained parents of a clause
    pub fn parents_clause_iter(&self, clause: ClauseIndex, state: &StateManager) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.clauses[clause.0].parents.iter(state)
    }
    
    /// Returns an iterator on the constrained children of a clause
    pub fn children_clause_iter(&self, clause: ClauseIndex, state: &StateManager) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.clauses[clause.0].children.iter(state)
    }
    
    /// Returns an iterator on all (constrained and unconstrained) the clauses of the graph
    pub fn clause_iter(&self) -> impl Iterator<Item = ClauseIndex> {
        (0..self.clauses.len()).map(ClauseIndex)
    }
    
    /// Returns an iterator over all the clauses (constrained and unconstrained) in which the variable is in the index
    pub fn variable_clause_body_iter(&self, variable: VariableIndex) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.variables[variable.0].clauses_body.iter().copied()
    }
    
    /// Returns an iterator over all the clauses (constrained and unconstrained) in which the variable is the head
    pub fn variable_clause_head_iter(&self, variable: VariableIndex) -> impl Iterator<Item = ClauseIndex> + '_ {
        self.variables[variable.0].clauses_head.iter().copied()
    }

    /// Returns an iterator over all the variable (fixed and not fixed) in a distribution
    pub fn distribution_variable_iter(&self, distribution: DistributionIndex) -> impl Iterator<Item = VariableIndex> {
        let first = self.distributions[distribution.0].first.0;
        let last = first + self.distributions[distribution.0].size;
        (first..last).map(VariableIndex)
    }

    /// Returns an iterator over all the variables, not fixed yet, in the body of the clause
    pub fn clause_body_iter(&self, clause: ClauseIndex, state: &StateManager) -> impl Iterator<Item = VariableIndex> + '_ {
        self.clauses[clause.0].body.iter(state)
    }
    
}

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
            assert_eq!(Some(1.0 / (i + 1) as f64), v.weight);
            assert_eq!(Some(DistributionIndex(i)), v.distribution);
            assert_eq!(vec_clause, v.clauses_head);
            assert_eq!(vec_clause, v.clauses_body);
            assert!(!state.get_bool(v.fixed));
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
            assert!(!state.get_bool(v.fixed));
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
        assert_eq!(vec![p[0], p[1]], clause.body.iter(&state).collect::<Vec<VariableIndex>>());
        assert!(state.get_bool(clause.constrained));
        assert_eq!(2, state.get_usize(clause.number_probabilistic));
        
        let c2 = g.add_clause(d[1], vec![d[0], p[2]], &mut state);
        let clause = &g.clauses[c2.0];
        assert_eq!(d[1], clause.head);
        assert_eq!(vec![d[0], p[2]], clause.body.iter(&state).collect::<Vec<VariableIndex>>());
        assert!(state.get_bool(clause.constrained));
        assert_eq!(1, state.get_usize(clause.number_probabilistic));
        
        let c3 = g.add_clause(d[0], vec![d[1]], &mut state);
        let clause = &g.clauses[c3.0];
        assert_eq!(d[0], clause.head);
        assert_eq!(vec![d[1]], clause.body.iter(&state).collect::<Vec<VariableIndex>>());
        assert!(state.get_bool(clause.constrained));
        assert_eq!(0, state.get_usize(clause.number_probabilistic));
    }
}