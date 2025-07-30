
use search_trail::StateManager;
use crate::core::problem::{Problem, ClauseIndex, VariableIndex};
use crate::common::Caching;

pub struct CachingScheme {
    strategy: Box<dyn CachingStrategy + Sync + Send>,
}

impl CachingScheme {

    pub fn new(caching: Caching) -> Self {
        let strategy: Box<dyn CachingStrategy + Sync + Send> = match caching {
            Caching::Hybrid => Box::<Hybrid>::default(),
            Caching::OmitBinary => Box::<OmitBinary>::default(),
            Caching::OmitImplicit => Box::<OmitImplicit>::default(),
        };
        Self { strategy }
    }

    pub fn get_key(&self, problem: &Problem, clauses: &[ClauseIndex], variables: &[VariableIndex], state: &StateManager) -> Vec<usize> {
        self.strategy.get_representation(problem, clauses, variables, state)
    }

    pub fn init(&mut self, number_clauses: usize, number_vars: usize) {
        self.strategy.init(number_clauses, number_vars);
    }
}

pub trait CachingStrategy {

    fn get_representation(&self, problem: &Problem, clauses: &[ClauseIndex], variables: &[VariableIndex], state: &StateManager) -> Vec<usize>;
    fn init(&mut self, number_clauses: usize, number_vars: usize);
}

#[derive(Default)]
pub struct Hybrid {}

impl CachingStrategy for Hybrid {
    
    fn get_representation(&self, _problem: &Problem, clauses: &[ClauseIndex], variables: &[VariableIndex], _state: &StateManager) -> Vec<usize> {
        let mut a = variables.iter().map(|v| v.0).collect::<Vec<usize>>();
        let mut b = clauses.iter().map(|c| c.0).collect::<Vec<usize>>();
        a.push(usize::MAX);
        a.append(&mut b);
        a
    }

    fn init(&mut self, _number_clauses: usize, _number_vars: usize) {
        
    }
}

#[derive(Default)]
pub struct OmitBinary {}

impl CachingStrategy for OmitBinary {

    fn get_representation(&self, problem: &Problem, clauses: &[ClauseIndex], variables: &[VariableIndex], _state: &StateManager) -> Vec<usize> {
        let mut a = variables.iter().map(|v| v.0).collect::<Vec<usize>>();
        let mut b = clauses.iter().copied().filter(|c| !problem[*c].is_binary()).map(|c| c.0).collect::<Vec<usize>>();
        a.push(usize::MAX);
        a.append(&mut b);
        a
    }

    fn init(&mut self, _number_clauses: usize, _number_vars: usize) {
        
    }

}

#[derive(Default)]
pub struct OmitImplicit {}

impl CachingStrategy for OmitImplicit {

    fn get_representation(&self, problem: &Problem, clauses: &[ClauseIndex], variables: &[VariableIndex], state: &StateManager) -> Vec<usize> {
        let mut a = variables.iter().map(|v| v.0).collect::<Vec<usize>>();
        let mut b = clauses.iter().copied().filter(|c| !problem[*c].is_binary() && problem[*c].is_modified(state)).map(|c| c.0).collect::<Vec<usize>>();
        a.push(usize::MAX);
        a.append(&mut b);
        a
    }

    fn init(&mut self, _number_clauses: usize, _number_vars: usize) {
        
    }

}
