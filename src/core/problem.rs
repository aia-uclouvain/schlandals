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

use std::cmp::Ordering;

use search_trail::*;
use super::literal::*;
use super::variable::*;
use super::clause::*;
use super::distribution::*;
use super::watched_vector::WatchedVector;
use super::bitvec::WORD_SIZE;

use rustc_hash::FxHashMap;

/// Abstraction used as a typesafe way of retrieving a `Variable` in the `Problem` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VariableIndex(pub usize);

/// Abstraction used as a typesafe way of retrieving a `Clause` in the `Problem` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ClauseIndex(pub usize);

/// Abstraction used as a typesafe way of retrieving a `Distribution` in the `Problem` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DistributionIndex(pub usize);

/// Abstraction used as a typesafe way of retrieving a watched vector in the `Problem` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct WatchedVectorIndex(pub usize);

/// Data structure representing the Problem.
#[derive(Debug)]
pub struct Problem {
    /// Vector containing the nodes of the problem
    variables: Vec<Variable>,
    /// Vector containing the clauses of the problem
    clauses: Vec<Clause>,
    /// Vector containing the distributions of the problem
    distributions: Vec<Distribution>,
    /// Store for each variables the clauses it watches
    watchers: Vec<Vec<ClauseIndex>>,
    /// Number of clauses in the problem
    number_clauses_problem: usize,
}

impl Problem {
    
    // --- PROBLEM CREATION --- //

    /// Creates a new empty implication problem
    pub fn new(state: &mut StateManager, n_var: usize, n_clause: usize) -> Self {
        let variables = (0..n_var).map(|i| Variable::new(i, None, None, state)).collect();
        let watchers = (0..n_var).map(|_| vec![]).collect();
        Self {
            variables,
            clauses: vec![],
            watchers,
            distributions: vec![],
            number_clauses_problem: n_clause,
        }
    }

    /// Add a distribution to the problem. In this case, a distribution is a set of probabilistic
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
                self.variables[new_index].set_old_index(initial_index);
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

        if is_learned {
            for i in 0..number_deterministic.min(2) {
                let variable = literal_vector[i].to_variable();
                self.watchers[variable.0].push(cid);
            }
            
            for i in 0..number_probabilistic.min(2) {
                let variable = literal_vector[number_deterministic + i].to_variable();
                self.watchers[variable.0].push(cid);
            }
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
                        self[child].increment_in_degree();
                    }
                }
            } else {
                self[variable].add_clause_negative_occurence(cid);
                if !is_learned {
                    for parent in self[variable].iter_clauses_positive_occurence().collect::<Vec<ClauseIndex>>() {
                        clause.add_parent(parent, state);
                        self[parent].add_child(cid, state);
                        clause.increment_in_degree();
                    }
                }
            }
            if let Some(distribution) = self[literal.to_variable()].distribution() {
                if !is_learned {
                    self[distribution].add_clause(cid, state);
                }
            }
        }
        self.clauses.push(clause);
        cid
    }


    /// Clears unnecessary space for the problem. After pre-processing, some variables, clauses or
    /// distributions might not be used anymore, we can remove them from the representation.
    pub fn clear_after_preprocess(&mut self, state: &mut StateManager) {
        // First, we delete the clauses 
        let mut clauses_map: FxHashMap<ClauseIndex, ClauseIndex> = FxHashMap::default();
        let mut new_clause_index = 0;
        for (i, clause) in self.clauses_iter().enumerate() {
            if self[clause].is_constrained(state) {
                clauses_map.insert(clause, ClauseIndex(new_clause_index));
                self.clauses.swap(i, new_clause_index);
                new_clause_index += 1;
            }
        }
        self.clauses.truncate(new_clause_index);
        self.clauses.shrink_to_fit();

        if self.clauses.is_empty() {
            return;
        }

        self.number_clauses_problem = self.clauses.len();

        for clause in self.clauses_iter() {
            self[clause].clear(&clauses_map, state);
        }

        let mut distributions_map: FxHashMap<DistributionIndex, DistributionIndex> = FxHashMap::default();
        let mut new_distribution_index = 0;
        for (i, distribution) in self.distributions_iter().enumerate() {
            if self[distribution].is_constrained(state) || self[distribution].number_unfixed(state) <= 1 {
                let new_size = self[distribution].number_unfixed(state);
                self[distribution].set_size(new_size);
                distributions_map.insert(distribution, DistributionIndex(new_distribution_index));
                self.distributions.swap(i, new_distribution_index);
                new_distribution_index += 1;
            }
        }
        self.distributions.truncate(new_distribution_index);
        self.distributions.shrink_to_fit();

        let mut variables_map: FxHashMap<VariableIndex, VariableIndex> = FxHashMap::default();
        let mut new_variable_index = 0;
        for (i, variable) in self.variables_iter().enumerate() {
            let number_remaining_clauses = self[variable].clear_clauses(&clauses_map);
            let is_unconstrained = if self[variable].is_probabilitic() {
                let distribution = self[variable].distribution().unwrap();
                (number_remaining_clauses == 0 && distributions_map.get(&distribution).is_none()) || self[variable].is_fixed(state)
            } else {
                number_remaining_clauses == 0 || self[variable].is_fixed(state)
            };
            if is_unconstrained {
                continue;
            }
            if self[variable].is_probabilitic() {
                let old_distribution = self[variable].distribution().unwrap();
                let new_distribution = distributions_map.get(&old_distribution).copied().unwrap();
                self[variable].set_distribution(new_distribution);
            }
            variables_map.insert(variable, VariableIndex(new_variable_index));
            self.variables.swap(i, new_variable_index);
            self.watchers.swap(i, new_variable_index);
            new_variable_index += 1;
        }

        self.variables.truncate(new_variable_index);
        self.variables.shrink_to_fit();
        self.watchers.truncate(new_variable_index);
        self.watchers.shrink_to_fit();

        let mut current_distribution = self.variables[0].distribution().unwrap();
        self[current_distribution].set_start(VariableIndex(0));
        for variable in self.variables_iter() {
            if let Some(distribution) = self[variable].distribution() {
                if distribution != current_distribution {
                    self[distribution].set_start(variable);
                    current_distribution = distribution;
                }
            }
        }

        for variable in self.variables_iter() {
            let word_index = variable.0 / WORD_SIZE;
            let mask: u128 = 1 << (variable.0 % WORD_SIZE);
            self[variable].set_bitmask(mask);
            self[variable].set_bitword_index(word_index);
        }

        for clause in self.clauses_iter() {
            self[clause].clear_literals(&variables_map);
            for option_v in self[clause].get_watchers() {
                if let Some(v) = option_v {
                    self.watchers[v.0].push(clause);
                }
            }
            let i = clause.0 + self.variables.len();
            let word_index = i / WORD_SIZE;
            let mask: u128 = 1 << (i % WORD_SIZE);
            self[clause].set_bitmask(mask);
            self[clause].set_bitword_index(word_index);
        }

        for distribution in self.distributions_iter() {
            self[distribution].update_clauses(&clauses_map, state);
        }
    }
    
    // --- problem MODIFICATIONS --- //
    
    /// Sets a variable to true or false.
    ///     - If true, Removes the variable from the body of the constrained clauses
    ///     - If false, and probabilistic, increase the counter of false variable in the distribution
    /// If the variable is the min or max variable not fixed, update the boundaries accordingly.
    pub fn set_variable(&mut self, variable: VariableIndex, value: bool, level: isize, reason: Option<Reason>, state: &mut StateManager) {
        self[variable].set_value(value, state);
        self[variable].set_decision_level(level);
        self[variable].set_reason(reason, state);

        // If probabilistic and false, update the counter
        if !value && self[variable].is_probabilitic() {
            let distribution = self[variable].distribution().unwrap();
            self[distribution].increment_number_false(state);
            self[distribution].remove_probability_mass(self[variable].weight().unwrap(), state);
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

    // --- QUERIES --- //
    
    /// Set a clause as unconstrained
    pub fn set_clause_unconstrained(&self, clause: ClauseIndex, state: &mut StateManager) {
        self[clause].set_unconstrained(state);
    }
    
    // --- GETTERS --- //
    
    /// Returns the number of clause in the problem
    pub fn number_clauses(&self) -> usize {
        self.clauses.len()
    }

    pub fn last_clause_subproblem(&self) -> ClauseIndex {
        ClauseIndex(self.clauses.len() - 1)
    }

    /// Returns the number of unlearned clauses (i.e., the number of clauses in the initial problem
    pub fn number_clauses_problem(&self) -> usize {
        self.number_clauses_problem
    }

    /// Returns the number of variable in the problem
    pub fn number_variables(&self) -> usize {
        self.variables.len()
    }
    
    /// Returns the number of distribution in the problem
    pub fn number_distributions(&self) -> usize {
        self.distributions.len()
    }
    
    // --- ITERATORS --- //
    
    /// Returns an iterator on all (constrained and unconstrained) the clauses of the problem
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

// --- Indexing the problem with the various indexes --- //

impl std::ops::Index<VariableIndex> for Problem {
    type Output = Variable;

    fn index(&self, index: VariableIndex) -> &Self::Output {
        &self.variables[index.0]
    }
}

impl std::ops::IndexMut<VariableIndex> for Problem {
    fn index_mut(&mut self, index: VariableIndex) -> &mut Self::Output {
        &mut self.variables[index.0]
    }
}

impl std::ops::Index<ClauseIndex> for Problem {
    type Output = Clause;

    fn index(&self, index: ClauseIndex) -> &Self::Output {
        &self.clauses[index.0]
    }
}

impl std::ops::IndexMut<ClauseIndex> for Problem {
    fn index_mut(&mut self, index: ClauseIndex) -> &mut Self::Output {
        &mut self.clauses[index.0]
    }
}


impl std::ops::Index<DistributionIndex> for Problem {
    type Output = Distribution;

    fn index(&self, index: DistributionIndex) -> &Self::Output {
        &self.distributions[index.0]
    }
}

impl std::ops::IndexMut<DistributionIndex> for Problem {
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
mod test_problem_creation {
    
    use crate::core::problem::*;
    use search_trail::StateManager;

    #[test]
    pub fn variables_creation() {
        let mut state = StateManager::default();
        let mut g = Problem::new(&mut state);
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
        let mut g = Problem::new(&mut state);
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
        let mut g = Problem::new(&mut state);
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
mod problem_update {
    
    use crate::core::problem::*;
    use search_trail::StateManager;
    
    fn get_problem(state: &mut StateManager) -> Problem {
        // Structure of the problem:
        //
        // c0 --> c2 --> c3
        //        ^       |
        //        |       v
        // c1 ----------> c4
        let mut g = Problem::new(state);
        let nodes: Vec<VariableIndex> = (0..10).map(|_| g.add_variable(false, None, None, state)).collect();
        let _c0 = g.add_clause(nodes[2], vec![nodes[0], nodes[1]], state);
        let _c1 = g.add_clause(nodes[5], vec![nodes[3], nodes[4]], state);
        let _c2 = g.add_clause(nodes[6], vec![nodes[2], nodes[5]], state);
        let _c3 = g.add_clause(nodes[8], vec![nodes[6], nodes[7]], state);
        let _c4 = g.add_clause(nodes[9], vec![nodes[5], nodes[8]], state);
        g
    }
    
    fn check_parents(problem: &Problem, clause: ClauseIndex, mut expected_parents: Vec<usize>, state: &StateManager) {
        let mut actual = problem.parents_clause_iter(clause, state).map(|c| c.0).collect::<Vec<usize>>();
        actual.sort();
        expected_parents.sort();
        assert_eq!(expected_parents, actual);
    }
    
    fn check_children(problem: &Problem, clause: ClauseIndex, mut expected_children: Vec<usize>, state: &StateManager) {
        let mut actual = problem.children_clause_iter(clause, state).map(|c| c.0).collect::<Vec<usize>>();
        actual.sort();
        expected_children.sort();
        assert_eq!(expected_children, actual);
    }
    
    #[test]
    fn remove_clause() {
        let mut state = StateManager::default();
        let mut g = get_problem(&mut state);
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

    use crate::core::problem::*;
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
        let mut problem = Problem::new(&mut state);
        check_bit_array(&vec![], &problem.variables_bit, &state);
        for i in 0..300 {
            problem.add_variable(false, None, None, &mut state);
            if i % 64 == 0 {
                check_bit_array(&vec![!0; (i / 64)+1], &problem.variables_bit, &state);
            }
        }
        check_bit_array(&vec![!0; 5], &problem.variables_bit, &state);

        check_bit_array(&vec![], &problem.clauses_bit, &state);
        for i in 0..300 {
            problem.add_clause(VariableIndex(i), vec![VariableIndex(i)], &mut state);
            if i % 64 == 0 {
                check_bit_array(&vec![!0; (i / 64)+1], &problem.clauses_bit, &state);
            }
        }
        check_bit_array(&vec![!0; 5], &problem.clauses_bit, &state);
    }
    
    fn deactivate_bit(bit_array: &mut Vec<u64>, word_index: usize, bit_index: usize) {
        bit_array[word_index] &= !(1 << bit_index);
    }
    
    #[test]
    fn update_representation() {
        let mut state = StateManager::default();
        let mut problem = Problem::new(&mut state);
        for _ in 0..300 {
            problem.add_variable(false, None, None, &mut state);
        }
        for i in 0..300 {
            problem.add_clause(VariableIndex(i), vec![VariableIndex(i)], &mut state);
        }
        let mut expected = vec![!0; 5];

        problem.set_variable(VariableIndex(0), false, 0, None, &mut state);
        problem.set_clause_unconstrained(ClauseIndex(0), &mut state);
        deactivate_bit(&mut expected, 0, 0);
        check_bit_array(&expected, &problem.variables_bit, &state);
        check_bit_array(&expected, &problem.clauses_bit, &state);

        problem.set_variable(VariableIndex(130), false, 0, None, &mut state);
        problem.set_clause_unconstrained(ClauseIndex(130), &mut state);
        deactivate_bit(&mut expected, 2, 2);
        check_bit_array(&expected, &problem.variables_bit, &state);
        check_bit_array(&expected, &problem.clauses_bit, &state);

        problem.set_variable(VariableIndex(131), false, 0, None, &mut state);
        problem.set_clause_unconstrained(ClauseIndex(131), &mut state);
        deactivate_bit(&mut expected, 2, 3);
        check_bit_array(&expected, &problem.variables_bit, &state);
        check_bit_array(&expected, &problem.clauses_bit, &state);

        problem.set_variable(VariableIndex(299), false, 0, None, &mut state);
        problem.set_clause_unconstrained(ClauseIndex(299), &mut state);
        deactivate_bit(&mut expected, 4, 43);
        check_bit_array(&expected, &problem.variables_bit, &state);
        check_bit_array(&expected, &problem.clauses_bit, &state);
    }
}
*/
