use search_trail::*;
use super::literal::*;
use super::variable::*;
use super::clause::*;
use super::distribution::*;
use malachite::rational::Rational;

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
        let variables = (0..n_var).map(|i| Variable::new(i, None, None, None, state)).collect();
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
    pub fn add_distributions(&mut self, distributions: &[Vec<Rational>], state: &mut StateManager) -> FxHashMap<usize, usize> {
        let mut mapping: FxHashMap<usize, usize> = FxHashMap::default();
        let mut current_start = 0;
        for (d_id, weights) in distributions.iter().enumerate() {
            let distribution = Distribution::new(d_id, VariableIndex(current_start), weights.len(), state);
            let distribution_id = DistributionIndex(self.distributions.len());
            self.distributions.push(distribution);
            let mut weight_with_ids = weights.iter().enumerate().map(|(i, w)| (w.clone(),i)).collect::<Vec<(Rational, usize)>>();
            weight_with_ids.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
            // j is the new index while i is the initial index
            // So the variable will be store at current_start + j instead of current_start + i (+ 1
            // for the mapping because in the input file indexes start at 1)
            for (j, (w, i)) in weight_with_ids.iter().enumerate() {
                let new_index = current_start + j;
                let initial_index = current_start + i;
                self.variables[new_index].set_distribution(distribution_id);
                self.variables[new_index].set_weight(w.clone());
                self.variables[new_index].set_distribution_index(*i);
                mapping.insert(initial_index + 1, new_index + 1);
            }
            current_start += weights.len();
        }
        mapping
    }
    
    pub fn add_clause(
        &mut self,
        literals: Vec<Literal>,
        state: &mut StateManager,
        is_learned: bool,
    ) -> ClauseIndex {
        let cid = ClauseIndex(self.clauses.len());
        for variable in literals.iter().take(2).map(|l| l.to_variable()) {
            self.watchers[variable.0].push(cid);
        }

        let number_deterministic_in_body = literals.iter().copied().filter(|l| !l.is_positive() && !self[l.to_variable()].is_probabilitic()).count();
        let mut clause = Clause::new(cid.0, literals, number_deterministic_in_body, is_learned, state);
        for literal in clause.iter().collect::<Vec<Literal>>() {
            let variable = literal.to_variable();
            if literal.is_positive() {
                self[variable].add_clause_positive_occurence(cid, state);
                if self[variable].is_probabilitic() {
                    clause.set_head_f_reachable(state);
                }
            } else {
                self[variable].add_clause_negative_occurence(cid, state);
            }
            if let Some(distribution) = self[literal.to_variable()].distribution() {
                self[distribution].add_clause(cid, state);
            }
        }
        // If the clause is not learned, we need to link it to the other clauses for FT-reachable propagation.
        if !is_learned {
            for literal in clause.iter().collect::<Vec<Literal>>() {
                let variable = literal.to_variable();
                if literal.is_positive() {
                    for child in self[variable].iter_clauses_negative_occurence(state).filter(|c| c.0 != self.clauses.len()).collect::<Vec<ClauseIndex>>() {
                        clause.add_child(child, state);
                        self[child].add_parent(cid, state);
                    }
                } else {
                    for parent in self[variable].iter_clauses_positive_occurence(state).filter(|c| c.0 != self.clauses.len()).collect::<Vec<ClauseIndex>>() {
                        clause.add_parent(parent, state);
                        self[parent].add_child(cid, state);
                    }
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
            if self[clause].is_active(state) {
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
            if self[distribution].is_constrained(state) {
                let new_size = self[distribution].iter_variables().filter(|v| !self[*v].is_fixed(state)).count();
                self[distribution].set_domain_size(new_size);
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
            self.watchers[i].clear();
            let number_remaining_clauses = self[variable].clear_clauses(&clauses_map, state);
            let is_unconstrained = if self[variable].is_probabilitic() {
                let distribution = self[variable].distribution().unwrap();
                (number_remaining_clauses == 0 && !distributions_map.contains_key(&distribution)) || self[variable].is_fixed(state)
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

        for clause in self.clauses_iter() {
            self[clause].clear_literals(&variables_map);
            for v in self[clause].get_watchers() {
                self.watchers[v.0].push(clause);
            }
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
    pub fn deactivate_clause(&self, clause: ClauseIndex, state: &mut StateManager) {
        self[clause].deactivate(state);
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
    pub fn clauses_iter(&self) -> impl Iterator<Item = ClauseIndex> + use<> {
        (0..self.clauses.len()).map(ClauseIndex)
    }

    /// Returns an iterator on the distributions of the problem
    pub fn distributions_iter(&self) -> impl Iterator<Item = DistributionIndex> + use<> {
        (0..self.distributions.len()).map(DistributionIndex)
    }
    
    pub fn variables_iter(&self) -> impl Iterator<Item = VariableIndex> + use<> {
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
