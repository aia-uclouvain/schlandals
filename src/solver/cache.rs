//Schlandalhttps://www.youtube.com/watch?v=-9lrYoX2cMks
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

use rustc_hash::{FxHashMap, FxHashSet};
use search_trail::StateManager;
use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::{Graph, VariableIndex, ClauseIndex};
use rug::Float;
pub struct SubProblem {
    start: usize,
    end: usize,
    probability: Option<Float>,
}

const WORD_SIZE: usize = 64;

pub struct Cache {
    /// Number of words required for the encoding of the clauses
    clause_offset: usize,
    /// Actual words representing the problems
    bits: Vec<u64>,
    /// Index to insert the next subproblem
    pub current: usize,
    init_size: usize,
    /// Maximum index in the memory arena
    limit: usize,
    hash_idx: FxHashMap<u64, Vec<SubProblem>>,
}

impl Cache {
    /// Creates a new cache with max_cache_size capacity, in MB
    pub fn new(max_cache_size: u64, g: &Graph) -> Self {
        // init size of 1Gb
        let init_size = 1_000_000_000 / (WORD_SIZE/8);
        // Convert max_cache_size in bytes
        let limit = ((max_cache_size*1_000_000) / (WORD_SIZE/8) as u64) as usize;
        let bits: Vec<u64> = vec![0;init_size];
        let clause_offset = (g.number_clauses() as f64 / WORD_SIZE as f64).ceil() as usize;
        Self {
            clause_offset,
            bits,
            current: 0,
            init_size,
            limit,
            hash_idx: FxHashMap::default(),
        }
    }
    
    fn same_subproblem(&self, p1: &SubProblem, start: usize, end: usize) -> bool {
        if p1.end - p1.start + 1 != end - start + 1 {
            return false;
        } else {
            for i in 0..(end - start + 1) {
                if self.bits[p1.start + i] != self.bits[start + i] {
                    return false;
                }
            }
        }
        true
    }
    
    fn get_sol_from_hash(&self, hash: u64, start: usize, end: usize) -> Option<Float> {
        match self.hash_idx.get(&hash) {
            None => None,
            Some(v) => {
                for sp in v {
                    if self.same_subproblem(sp, start, end) {
                        return sp.probability.clone();
                    }
                }
                None
            }
        }
    }
    
    fn resize(&mut self) -> usize {
        let cur_size = self.bits.len();
        if cur_size >= self.limit {
            self.bits.clear();
            self.bits.resize(self.init_size, 0);
            0
        } else if 2*cur_size >= self.limit {
            self.bits.resize(self.limit, 0);
            cur_size
        } else {
            self.bits.resize(2*cur_size, 0);
            cur_size
        }
    }
    
    pub fn get(&mut self, extractor: &ComponentExtractor, component: ComponentIndex, graph: &Graph, state: &StateManager) -> (Option<Float>, (u64, usize)) {
        if self.current + self.clause_offset >= self.bits.len() {
            self.current = self.resize();
        }
        let mut seen_vars: FxHashSet<VariableIndex> = FxHashSet::default();
        let mut count_var = 0;
        let mut nb_var_words = 0;
        let mut hash = 0u64;
        let mut cls = extractor.component_iter(component).collect::<Vec<ClauseIndex>>();
        cls.sort();
        for clause in cls {
            self.bits[self.current + (clause.0 / WORD_SIZE)] |= 1 << (clause.0 % WORD_SIZE);
            let head = graph.get_clause_head(clause);
            if !seen_vars.contains(&head) {
                seen_vars.insert(head);
                if !graph.is_variable_bound(head, state) {
                    self.bits[self.current + self.clause_offset + nb_var_words] |= 1 << (count_var % WORD_SIZE);
                }
                count_var += 1;
                if count_var % WORD_SIZE == 0 {
                    nb_var_words += 1;
                    if self.bits.len() == self.current + self.clause_offset + nb_var_words {
                        let new_index = self.resize();
                        if new_index == 0 {
                            return self.get(extractor, component, graph, state);
                        }
                    }
                    hash ^= self.bits[self.current + self.clause_offset + (count_var / WORD_SIZE) - 1];
                }
            }
            for var in graph.clause_body_iter(clause) {
                if !seen_vars.contains(&var) {
                    seen_vars.insert(var);
                    if !graph.is_variable_bound(var, state) {
                        self.bits[self.current + self.clause_offset + (count_var / WORD_SIZE)] |= 1 << (count_var % WORD_SIZE);
                    }
                    count_var += 1;
                    if count_var % WORD_SIZE == 0 {
                        nb_var_words += 1;
                        if self.bits.len() == self.current + self.clause_offset + nb_var_words {
                            let new_index = self.resize();
                            if new_index == 0 {
                                return self.get(extractor, component, graph, state);
                            }
                        }
                        hash ^= self.bits[self.current + self.clause_offset + (count_var / WORD_SIZE) - 1];
                    }
                }
            }
        }
        for i in 0..self.clause_offset {
            hash ^= self.bits[self.current + i];
        }
        let comp_size = self.clause_offset + nb_var_words;
        let sol = self.get_sol_from_hash(hash, self.current, self.current + comp_size);
        if sol.is_some() {
            for i in 0..comp_size {
                self.bits[self.current + i] = 0;
            }
        } else {
            self.hash_idx.insert(hash, vec![SubProblem {
                start: self.current,
                end: self.current + comp_size,
                probability: None
            }]);
            self.current += comp_size;
        }
        (sol, (hash, self.current))
    }
    
    pub fn set(&mut self, hash: u64, start: usize, probability: Float) {
        let sps = self.hash_idx.get_mut(&hash).unwrap();
        for sp in sps.iter_mut() {
            if sp.start == start {
                sp.probability = Some(probability.clone());
            }
        }
    }
}
