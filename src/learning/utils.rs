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
use std::path::PathBuf;
use rayon::prelude::*;
use crate::branching::*;
use crate::Branching;
use search_trail::StateManager;
use crate::core::components::ComponentExtractor;
use crate::propagator::Propagator;
use crate::parser::*;
use crate::solvers::*;
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

/// Generates a vector of optional Dacs from a list of input files
pub fn generate_dacs<R>(inputs: Vec<PathBuf>, branching: Branching, epsilon: f64) -> Vec<Option<Dac<R>>>
    where R: SemiRing
{
    inputs.par_iter().map(|input| {
        // We compile the input. This can either be a .cnf file or a fdac file.
        // If the file is a fdac file, then we read directly from it
        match type_of_input(input) {
            FileType::CNF => {
                println!("Compiling {}", input.to_str().unwrap());
                // The input is a CNF file, we need to compile it from scratch
                let compiler = make_solver!(input, branching, epsilon, None, false);
                compile!(compiler)
            },
            FileType::FDAC => {
                println!("Reading {}", input.to_str().unwrap());
                // The query has already been compiled, we just read from the file.
                Some(Dac::from_file(input))
            }
        }
    }).collect::<Vec<_>>()
}

/// Decides whether early stopping should be performed or not
pub fn do_early_stopping(avg_loss:f64, prev_loss:f64, count:&mut usize, stopping_criterion:f64, patience:usize, delta:f64) -> bool {
    if (avg_loss-prev_loss).abs()<delta {
        *count += 1;
    }
    else {
        *count = 0;
    }
    if (avg_loss < stopping_criterion) || *count>=patience {
        true
    }
    else {
        false
    }
}

/// Structure representing a dataset for the learners. A dataset is a set of queries (boolean
/// formulas compiled into an arithmetic circuit) associated with an expected probability
pub struct Dataset<R> 
    where R: SemiRing
{
    queries: Vec<Dac<R>>,
    expected: Vec<R>,
}

impl<R> Dataset<R>
    where R: SemiRing
{

    /// Creates a new dataset from the provided queries and expected probabilities
    pub fn new(queries: Vec<Dac<R>>, expected: Vec<R>) -> Self {
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
    pub fn expected(&self, query_index: usize) -> &R {
        &self.expected[query_index]
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
