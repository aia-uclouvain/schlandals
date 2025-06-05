use search_trail::StateManager;

use crate::core::components::{ComponentExtractor, ComponentIndex};
use crate::core::problem::*;
use crate::propagator::Propagator;
use malachite::rational::Rational;
use crate::common::rational;

pub struct Preprocessor<'b>
{
    /// Implication problem of the input CNF formula
    problem: &'b mut Problem,
    /// State manager that allows to retrieve previous values when backtracking in the search tree
    state: &'b mut StateManager,
    /// The propagator
    propagator: &'b mut Propagator,
    /// component extractor
    component_extractor: &'b mut ComponentExtractor,
}

impl<'b> Preprocessor<'b>
where
{

    pub fn new(problem: &'b mut Problem, state: &'b mut StateManager, propagator: &'b mut Propagator, component_extractor: &'b mut ComponentExtractor) -> Self {
        Self {
            problem,
            state,
            propagator,
            component_extractor,
        }
    }
    
    pub fn preprocess(&mut self) -> Option<Rational> {
        let mut p = rational(1.0);

        for variable in self.problem.variables_iter() {
            if self.problem[variable].is_probabilitic() && self.problem[variable].weight().unwrap() == 0.0 {
                self.propagator.add_to_propagation_stack(variable, false, 0, None);
            }
        }

        // Find unit clauses
        for clause in self.problem.clauses_iter() {
            if self.problem[clause].is_unit(self.state) {
                let l = self.problem[clause].get_unit_assigment(self.state);
                self.propagator.add_to_propagation_stack(l.to_variable(), l.is_positive(), 0, None);
            }
        }

        for l in self.problem.clauses_iter().filter(|c| self.problem[*c].is_unit(self.state)).map(|c| self.problem[c].get_unit_assigment(self.state)) {
            self.propagator.add_to_propagation_stack(l.to_variable(), l.is_positive(), 0, None);
        }
        
        match self.propagator.propagate(self.problem, self.state, ComponentIndex(0), self.component_extractor, 0) {
            Err(_) => return None,
            Ok(_) => {
                p *= self.propagator.get_propagation_prob();
            }
        };
        Some(p)
    }
}
