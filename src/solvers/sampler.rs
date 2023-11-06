
//Schlandal
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

use rug::Float;
use rustc_hash::FxHashMap;
use search_trail::{StateManager, SaveAndRestore};

use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::graph::*;
use crate::core::literal::Literal;
use crate::core::variable::Reason;
use crate::branching::BranchingDecision;
use crate::preprocess::Preprocessor;
use crate::propagator::Propagator;
use super::statistics::Statistics;
use crate::common::*;
use crate::PEAK_ALLOC;

use super::{Bounds, ProblemSolution, Unsat};


pub struct SamplerSolver<'b, B, const S: bool>
where 
    B: BranchingDecision + ?Sized,
{
    /// Implication graph of the input CNF formula
    graph: Graph,
    /// State manager that allows to retrieve previous values when backtracking in the search tree
    state: StateManager,
    /// Extracts the connected components in the graph
    component_extractor: ComponentExtractor,
    /// Heuristics that decide on which distribution to branch next
    branching_heuristic: &'b mut B,
    /// The propagator
    propagator: Propagator,
    /// Cache used to store results of sub-problems
    cache: FxHashMap<CacheEntry, Bounds>,
    /// Statistics collectors
    statistics: Statistics<S>,
    /// Memory limit allowed for the solver. This is a global memory limit, not a cache-size limit
    mlimit: u64,
}

impl<'b, B, const S: bool> SamplerSolver<'b, B, S>
where
    B: BranchingDecision + ?Sized
{
    pub fn new(
        graph: Graph,
        state: StateManager,
        component_extractor: ComponentExtractor,
        branching_heuristic: &'b mut B,
        propagator: Propagator,
        mlimit: u64,
    ) -> Self {
        let cache = FxHashMap::default();
        Self {
            graph,
            state,
            component_extractor,
            branching_heuristic,
            propagator,
            cache,
            statistics: Statistics::default(),
            mlimit,
        }
    }

    fn restore(&mut self) {
        self.propagator.restore(&self.state);
        self.state.restore_state();
    }
    
    fn sample_distribution(&self, distribution: DistributionIndex) -> VariableIndex {
        debug_assert!(self.graph[distribution].is_constrained(&self.state));
        let proba_budget: f64 = self.graph[distribution].iter_variables().filter(|v| !self.graph[*v].is_fixed(&self.state)).map(|v| self.graph[v].weight().unwrap()).sum();
        let mut target_p: f64 = rand::random::<f64>()*proba_budget;
        debug_assert!(0.0 <= target_p && target_p <= proba_budget);
        for variable in self.graph[distribution].iter_variables() {
            if !self.graph[variable].is_fixed(&self.state) {
                let weight = self.graph[variable].weight().unwrap();
                target_p -= weight;
                if target_p <= 0.0 {
                    return variable;
                }
            }
        }
        // Should never happen
        panic!("No suitable variable found for sampling (target p {} budget {})", target_p, proba_budget);
    }
    
    pub fn sample(&mut self) -> ProblemSolution {
        let mut bounds: Vec<Bounds> = vec![];
        self.propagator.init(self.graph.number_clauses());
        let preproc = Preprocessor::new(&mut self.graph, &mut self.state, self.branching_heuristic, &mut self.propagator, &mut self.component_extractor).preprocess(false);
        if preproc.is_none() {
            return ProblemSolution::Err(Unsat);
        }
        let w_factor = self.propagator.get_propagation_prob().clone();
        if self.graph.distributions_iter().find(|d| self.graph[*d].is_constrained(&self.state)).is_none() {
            return ProblemSolution::Ok(w_factor);
        }
        let mut sample: Vec<Literal> = vec![];
        let mut p_in = f128!(0.0);
        let mut p_out = f128!(0.0);
        for _ in 0..1000 {
            let mut sample_p = w_factor.clone();
            let mut trail_size = 0;
            let mut is_model = true;
            for distribution in self.graph.distributions_iter() {
                if self.graph[distribution].is_constrained(&self.state) {
                    let variable = self.sample_distribution(distribution);
                    let v_weight = self.graph[variable].weight().unwrap();
                    self.state.save_state();
                    trail_size += 1;
                    match self.propagator.propagate_variable(variable, true, &mut self.graph, &mut self.state, ComponentIndex(0), &mut self.component_extractor, 0) {
                        Err((_, reason)) => {
                            is_model = false;
                            sample_p *= v_weight;
                            if let Some(r) = reason {
                                match r {
                                    Reason::Clause(c) => {
                                        if !self.graph[c].is_learned() {
                                            p_out += sample_p.clone();
                                        }
                                    },
                                    Reason::Distribution(_) => {
                                        p_out += sample_p.clone();
                                    }
                                };
                            }
                            break;
                        },
                        Ok(_) => {
                            for literal in self.propagator.assignments_iter(&self.state) {
                                if self.graph[literal.to_variable()].is_probabilitic() && literal.is_positive() {
                                    sample.push(literal);
                                }
                            }
                            sample_p *= self.propagator.get_propagation_prob();
                        }
                    }
                }
            }
            
            for _ in 0..trail_size {
                self.restore();
            }
            if sample.is_empty() {
                continue;
            }
            if is_model {
                p_in += sample_p;
                let blocking_clause = sample.iter().map(|l| l.opposite()).collect::<Vec<Literal>>();
                let clause = self.graph.add_clause(blocking_clause, None, &mut self.state, true);
                self.component_extractor.add_clause_to_component(ComponentIndex(0), clause);
            }
            sample.clear();
            if p_in == 1.0 - p_out.clone() {
                bounds.push((p_in.clone(), 1 - p_out.clone()));
                break;
            } else {
                bounds.push((p_in.clone(), 1 - p_out.clone()));
            }
        }
        print!("{}", bounds.iter().map(|b| format!("{} {}", b.0, b.1)).collect::<Vec<String>>().join(" "));
        Ok(p_in)
    }

}
