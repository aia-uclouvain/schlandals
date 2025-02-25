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
use clap::ValueEnum;
use std::hash::Hash;
use malachite::Rational;
use malachite::num::conversion::traits::RoundingFrom;
use malachite::rounding_modes::RoundingMode::Nearest;

macro_rules! F128 {
    ($v:expr) => {
        Rational::try_from($v).unwrap()
    };
}
pub(crate) use F128;

macro_rules! rational_to_f64 {
    ($v:expr) => {
        f64::rounding_from($v, Nearest).0
    }
}
pub(crate) use rational_to_f64;

pub const FLOAT_CMP_THRESHOLD: f64 = 0.00000;

pub type Bounds = (Rational, Rational);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Branching {
    /// Minimum In-degree of a clause in the implication-graph
    MinInDegree,
    DLCS,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Loss {
    MAE,
    MSE,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Semiring {
    Probability,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Optimizer {
    Adam,
    SGD,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum ApproximateMethod {
    /// Bound-based pruning
    Bounds,
    /// Limited Discrepancy Search
    LDS,
}

impl std::fmt::Display for ApproximateMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApproximateMethod::Bounds => write!(f, "bounds"),
            ApproximateMethod::LDS => write!(f, "lds"),
        }
    }
}

/// A key of the cache. It is composed of
///     1. A hash representing the sub-problem being solved
///     2. The bitwise representation of the sub-problem being solved
/// 
/// We adopt this two-level representation for the cache key for efficiency reason. The hash is computed during
/// the detection of the components and is a XOR of random bit string. This is efficient but do not ensure that
/// two different sub-problems have different hash.
/// Hence, we also provide an unique representation of the sub-problem, using 64 bits words, in case of hash collision.
#[derive(Default, Clone)]
pub struct CacheKey {
    hash: u64,
    repr: String,
}

impl CacheKey {
    pub fn new(hash: u64, repr: String) -> Self {
        Self {
            hash,
            repr,
        }
    }
}

impl Hash for CacheKey {

    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }

}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            false
        } else {
            self.repr == other.repr
        }
    }
}

impl Eq for CacheKey {}

/// This structure represent a (possibly partial) solution found by the solver.
/// It is represented by a lower- and upper-bound on the true probability at the time at which the
/// solution was found.
#[derive(Clone)]
pub struct Solution {
    /// Lower bound on the true probability
    lower_bound: Rational,
    /// Upper bound on the true probability
    upper_bound: Rational,
    /// Number of seconds, since the start of the search, at which the solution was found
    time_found: u64,
}

impl Solution {

    pub fn new(lower_bound: Rational, upper_bound: Rational, time_found: u64) -> Self {
        Self {
            lower_bound,
            upper_bound,
            time_found,
        }
    }

    pub fn has_converged(&self, epsilon: f64) -> bool {
        let conv_factor = F128!((1.0 + epsilon + 0.0000001).powf(2.0));
        self.upper_bound <= self.lower_bound.clone()*conv_factor
    }

    pub fn print(&self) {
        println!("{}", self);
    }

    pub fn to_f64(&self) -> f64 {
        rational_to_f64!(self.lower_bound.clone() * self.upper_bound.clone()).sqrt()
    }

    pub fn bounds(&self) -> (f64, f64) {
        (rational_to_f64!(&self.lower_bound), rational_to_f64!(&self.upper_bound))
    }

    pub fn epsilon(&self) -> f64 {
        let (lb, ub) = self.bounds();
        if lb != 0.0 {
            (ub / lb ).sqrt() - 1.0
        } else {
            f64::MAX
        }
    }
}

impl std::fmt::Display for Solution {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (lb, ub) = self.bounds();
        write!(f, "Estimated probability {:.8} with bounds [{:.8} {:.8}] (epsilon {}) found in {} seconds", (lb * ub).sqrt(), lb, ub, self.epsilon(), self.time_found)
    }
}
