//Schlandals
//Copyright (C) 2022 A. Dubray, L. Dierckx
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
use crate::{common::f128, diagrams::{semiring::SemiRing, dac::dac::Dac}};

/// Calculates the softmax (the normalized exponential) function, which is a generalization of the
/// logistic function to multiple dimensions.
///
/// Takes in a vector of real numbers and normalizes it to a probability distribution such that
/// each of the components are in the interval (0, 1) and the components add up to 1. Larger input
/// components correspond to larger probabilities.
/// From https://docs.rs/compute/latest/src/compute/functions/statistical.rs.html#43-46
pub fn softmax(x: &[f64]) -> Vec<Float> {
    let sum_exp: f64 = x.iter().map(|i| i.exp()).sum();
    x.iter().map(|i| f128!(i.exp() / sum_exp)).collect()
}

/// Structure representing a dataset for the learners. A dataset is a set of queries (boolean
/// formulas compiled into an arithmetic circuit) associated with an expected probability
pub struct Dataset<R> 
    where R: SemiRing
{
    queries: Vec<Dac<R>>,
    expected: Vec<f64>,
}

impl<R> Dataset<R>
    where R: SemiRing
{

    /// Creates a new dataset from the provided queries and expected probabilities
    pub fn new(queries: Vec<Dac<R>>, expected: Vec<f64>) -> Self {
        Self {
            queries,
            expected,
        }
    }

    /// Returns size of the dataset
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    /// Returns a reference to the queries of the dataset
    pub fn get_queries(&self) -> &Vec<Dac<R>> {
        &self.queries
    }

    /// Returns a mutable reference to the queries of the dataset
    pub fn get_queries_mut(&mut self) -> &mut Vec<Dac<R>> {
        &mut self.queries
    }

    /// Returns the expected output for the required query
    pub fn expected(&self, query_index: usize) -> f64 {
        self.expected[query_index]
    }
}

impl<R: SemiRing + 'static> std::ops::Index<usize> for Dataset<R> {
    type Output = Dac<R>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.queries[index]
    }
}

impl<R: SemiRing + 'static> std::ops::IndexMut<usize> for Dataset<R> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.queries[index]
    }
}
