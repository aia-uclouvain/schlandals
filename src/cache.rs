use crate::solver::DistributionPartialDomain;
use crate::core::problem::{DistributionIndex,VariableIndex};
use crate::common::*;

use rustc_hash::FxHashMap;

/// An entry in the cache for the search. It contains the bounds computed when the sub-problem was
/// explored as well as various informations used by the solvers.
#[derive(Clone)]
pub struct CacheEntry {
    /// The current bounds on the sub-problem
    bounds: Bounds,
    /// Maximum discrepancy used for that node
    discrepancy: usize,
    /// The distribution on which to branch in this problem
    distribution: Option<DistributionIndex>,
    /// Cache entries corresponding to children in the search tree
    children: FxHashMap<VariableIndex, Option<Vec<usize>>>,
    /// Index of the entry's key in the vector of keys
    cache_key_index: usize,
    /// Is the entry complete (i.e., all the sub-tree has been explored)
    complete: bool,
    /// Sub-problem representation for AC compilation
    domains: Vec<DistributionPartialDomain>,
}

impl CacheEntry {

    /// Returns a new cache entry
    pub fn new(cache_key_index: usize, domains: Vec<DistributionPartialDomain>) -> Self {
        Self {
            bounds: (rational(0.0), rational(0.0)),
            discrepancy: 0,
            distribution: None,
            children: FxHashMap::default(),
            cache_key_index,
            complete: false,
            domains,
        }
    }

    pub fn distribution(&self) -> Option<DistributionIndex> {
        self.distribution
    }

    pub fn set_distribution(&mut self, distribution: Option<DistributionIndex>) {
        self.distribution = distribution;
    }

    pub fn discrepancy(&self) -> usize {
        self.discrepancy
    }

    pub fn set_discrepancy(&mut self, discrepancy: usize) {
        self.discrepancy = discrepancy;
    }

    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    pub fn set_bounds(&mut self, bounds: Bounds) {
        self.bounds = bounds;
    }

    pub fn cache_key_index(&self) -> usize {
        self.cache_key_index
    }

    pub fn number_children(&self) -> usize {
        self.children.len()
    }

    pub fn add_child(&mut self, key: VariableIndex, child: Option<Vec<usize>>) {
        self.children.insert(key, child);
    }

    pub fn children_variables(&self) -> Vec<VariableIndex> {
        self.children.keys().copied().collect()
    }

    pub fn child_keys(&self, variable: VariableIndex) -> Option<Vec<usize>> {
        self.children.get(&variable).unwrap().clone()
    }

    pub fn is_complete(&self) -> bool {
        self.complete
    }

    pub fn completed(&mut self) {
        self.complete = true;
    }

    pub fn domains(&self) -> &Vec<DistributionPartialDomain> {
        &self.domains
    }
}
