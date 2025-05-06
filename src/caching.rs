
use search_trail::StateManager;
use crate::core::problem::{Problem, ClauseIndex, VariableIndex};
use crate::common::Caching;
use std::hash::Hash;

pub struct CachingScheme {
    strategy: Box<dyn CachingStrategy + Sync + Send>,
    two_level_caching: bool,
}

impl CachingScheme {

    pub fn new(two_level_caching: bool, caching: Caching) -> Self {
        let strategy: Box<dyn CachingStrategy + Sync + Send> = match caching {
            Caching::Hybrid => Box::<Hybrid>::default(),
            Caching::OmitBinary => Box::<OmitBinary>::default(),
            Caching::OmitImplicit => Box::<OmitImplicit>::default(),
        };
        Self { strategy, two_level_caching }
    }

    pub fn get_key(&self, problem: &Problem, clauses: &[ClauseIndex], variables: &[VariableIndex], hash: u64, state: &StateManager) -> CacheKey {
        let repr = self.strategy.get_representation(problem, clauses, variables, state);
        CacheKey {
            hash,
            repr,
            two_level: self.two_level_caching,
        }
    }

    pub fn init(&mut self, number_clauses: usize, number_vars: usize) {
        self.strategy.init(number_clauses, number_vars);
    }
}

/// A key of the cache. It is composed of
///     1. A hash representing the sub-problem being solved
///     2. Therepresentation of the sub-problem being solved computed by the caching strategy
///
/// We adopt this two-level representation for the cache key for efficiency reason. The hash is computed during
/// the detection of the components and is a XOR of random bit string. This is efficient but do not ensure that
/// two different sub-problems have different hash.
/// Hence, we also provide an unique representation of the sub-problem, using 64 bits words, in case of hash collision.
#[derive(Default, Clone)]
pub struct CacheKey {
    hash: u64,
    repr: Vec<usize>,
    two_level: bool,
}

impl CacheKey {
    pub fn new(hash: u64, repr: Vec<usize>, two_level: bool) -> Self {
        Self {
            hash,
            repr,
            two_level,
        }
    }
}

impl Hash for CacheKey {

    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if self.two_level {
            self.hash.hash(state);
        } else {
            self.repr.hash(state);
        }
    }

}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.repr == other.repr
    }
}

impl Eq for CacheKey {}

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
