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

//! This module provides an implementation of a learner which aims at learning distribution 
//! parameters from a set of queries. The set of queries is expected to come from a same problem
//! and to share distributions. The expected value of each query should also be given in the input.
//! Each of the queries will be compiled in an arithmetic circuit (AC). The compilation can be 
//! partial (using approximations) or exact in function of the input parameters.
//! Once the queries are compiled, the distributions can be learned with the train function using
//! gradient descent optimization. Therefore, at each training epoch, the set of queries is evaluated
//! with the current values of the distributions. Then, the loss between the found values and the
//! expected ones is computed and used to derive the gradient value of each circuit parameter.
//! The values of the distributions are then updated using the found gradient values and the
//! process is repeated until convergence or until meeting a stopping criterion.
//!
//! Additionnaly, it is possible to provide a test set that will be evaluated once before the
//! training and a second time after the learning loop with the learned distribution values.
//!
//! Note that the 'learner' module is equivalent to this learner but this learner uses tensor automatic
//! backpropagation while 'learner' computes the gradients for floats.

use std::path::PathBuf;
use crate::diagrams::dac::dac::*;
use super::logger::Logger;
use crate::parser::*;
use crate::Branching;
use crate::{Optimizer as OptChoice, Loss};
use std::time::{Instant, Duration};
use rayon::prelude::*;
use super::Learning;
use super::utils::*;
use std::f64::consts::E;
use crate::diagrams::semiring::SemiRing;
use super::*;


use tch::{Kind, nn, Device, Tensor, IndexOp, Reduction};
use tch::nn::{OptimizerConfig, Adam, Sgd, Optimizer};


/// Abstraction used as a typesafe way of retrieving a `DAC` in the `Learner` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DacIndex(pub usize);

/// Structure used to learn the distribution parameters from a set of queries
pub struct TensorLearner<const S: bool>
{
    train: Dataset<Tensor>,
    test: Dataset<Tensor>,
    distribution_tensors: Vec<Tensor>,
    log: Logger<S>,
    epsilon: f64,
    optimizer: Optimizer,
    }

impl <const S: bool> TensorLearner<S>
{
    /// Creates a new learner for the inputs given. Each inputs represent a query that needs to be
    /// solved, and the expected_outputs contains, for each query, its expected probability.
    pub fn new(inputs: Vec<PathBuf>, mut expected_outputs:Vec<f64>, epsilon:f64, branching: Branching, outfolder: Option<PathBuf>, jobs:usize, timeout: u64, optimizer: OptChoice, test_inputs:Vec<PathBuf>, mut test_expected_o:Vec<f64>) -> Self {
        rayon::ThreadPoolBuilder::new().num_threads(jobs).build_global().unwrap();
        
        // Retrieves the distributions values and computes their unsoftmaxed values
        let distributions = distributions_from_cnf(&inputs[0]);
        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        // Retrieves which distributions are learned
        let learned_distributions = learned_distributions_from_cnf(&inputs[0]);
        let distribution_tensors = distributions.iter().enumerate().map(|(i, distribution)| {
            let t = Tensor::from_slice(&distribution.iter().map(|d| d.log(E)).collect::<Vec<f64>>()).requires_grad_(learned_distributions[i]);
            root.var_copy(&format!("Distribution {}", i+1), &t)
        }).collect::<Vec<Tensor>>();

        // Compiling the train and test queries into arithmetic circuits
        let mut train_dacs = generate_dacs(inputs, branching, epsilon, timeout);
        let mut test_dacs = generate_dacs(test_inputs, branching, epsilon, timeout);
        // Creating train and test datasets
        let mut train_data = vec![];
        let mut train_expected = vec![];
        let mut test_data = vec![];
        let mut test_expected = vec![];
        while !train_dacs.is_empty() {
            let d = train_dacs.pop().unwrap();
            let expected = expected_outputs.pop().unwrap();
            if let Some(d) = d {
                train_data.push(d);
                train_expected.push(Tensor::from_f64(expected));
            }
        }
        while !test_dacs.is_empty() {
            let d = test_dacs.pop().unwrap();
            let expected = test_expected_o.pop().unwrap();
            if let Some(d) = d {
                test_data.push(d);
                test_expected.push(Tensor::from_f64(expected));
            }
        }
        let train_dataset = Dataset::new(train_data, train_expected);
        let test_dataset = Dataset::new(test_data, test_expected);
        // Creating the optimizer
        let optimizer = match optimizer {
            OptChoice::Adam => Adam::default().build(&vs, 1e-4).unwrap(),
            OptChoice::SGD => Sgd::default().build(&vs, 1e-4).unwrap(),
        };
        // Initializing the logger
        let log = Logger::new(outfolder.as_ref(), train_dataset.len(), test_dataset.len());
        
        Self { 
            train: train_dataset,
            test: test_dataset,
            distribution_tensors,
            log,
            epsilon,
            optimizer,
        }
    }

    // --- Getters --- //

    /// Returns a double vector of tensors. Each entry (d,i) is a tensor representing the softmaxed
    /// version of the i-th value of vector d
    pub fn get_softmaxed_array(&self) -> Vec<Vec<Tensor>> {
        let softmaxed_distributions: Vec<Tensor> = self.distribution_tensors.iter().map(|tensor| tensor.softmax(-1, Kind::Float)).collect();
        softmaxed_distributions.iter().map(|tensor| {
            (0..tensor.size()[0]).map(|idx| {
                tensor.i(idx)
            }).collect::<Vec<Tensor>>()
        }).collect()
    }

    // --- Evaluation --- //

    // Evaluate the different train DACs and return the results
    fn evaluate(&mut self) -> Vec<f64> {
        let softmaxed = self.get_softmaxed_array();
        for dac in self.train.get_queries_mut() {
            dac.reset_distributions(&softmaxed);
        }
        self.train.get_queries_mut().par_iter_mut().for_each(|d| {
            d.evaluate();
        });
        self.train.get_queries().iter().map(|d| d.circuit_probability().to_f64()).collect()
    }

    // Evaluate the different test DACs and return the results
    fn test(&mut self) -> Vec<f64> {
        let softmaxed = self.get_softmaxed_array();
        for dac in self.test.get_queries_mut() {
            dac.reset_distributions(&softmaxed);
        }
        self.test.get_queries_mut().par_iter_mut().for_each(|d| {
            d.evaluate();
        });
        self.test.get_queries().iter().map(|d| d.circuit_probability().to_f64()).collect()
    }
}

impl<const S: bool> Learning for TensorLearner<S> {
    /// Training loop for the train dacs, using the given training parameters
    fn train(&mut self, params: &LearnParameters) {
        let mut prev_loss = 1.0;
        let mut count_no_improve = 0;
        self.log.start();
        let start = Instant::now();
        let to = Duration::from_secs(params.learning_timeout());

        // Evaluate the test set before training, if it exists
        if self.test.len()!=0{
            let predictions = self.test();
            let mut loss_epoch = Tensor::from(0.0);
            let mut test_loss = vec![0.0; self.test.len()];
            for i in 0..self.test.len() {
                let loss_i = match params.loss() {
                    Loss::MAE => self.test[i].circuit_probability().l1_loss(&self.test.expected(i), Reduction::Mean),
                    Loss::MSE => self.test[i].circuit_probability().mse_loss(&self.test.expected(i), Reduction::Mean),
                };
                test_loss[i] = loss_i.to_f64();
                loss_epoch += loss_i;
            }
            self.log.log_test(&test_loss, self.epsilon, &predictions);
        }

        // Training loop
        let mut train_loss = vec![0.0; self.train.len()];
        for e in 0..params.nepochs() {
            // Timeout
            if start.elapsed() > to { 
                break;
            }
            // Update the learning rate
            let learning_rate = params.lr() * params.lr_drop().powf(((1+e) as f64/ params.epoch_drop() as f64).floor());
            self.optimizer.set_lr(learning_rate);
            // Forward pass
            let predictions = self.evaluate();
            // Compute the loss
            let mut loss_epoch = Tensor::from(0.0);
            for i in 0..self.train.len() {
                let loss_i = match params.loss() {
                    Loss::MAE => self.train[i].circuit_probability().l1_loss(&self.train.expected(i), Reduction::Mean),
                    Loss::MSE => self.train[i].circuit_probability().mse_loss(&self.train.expected(i), Reduction::Mean),
                };
                train_loss[i] = loss_i.to_f64();
                loss_epoch += loss_i;
            }
            let avg_loss = train_loss.iter().sum::<f64>() / train_loss.len() as f64;
            // Compute the gradients and update the parameters
            self.optimizer.backward_step(&loss_epoch);
            // Log the epoch
            self.log.log_epoch(&train_loss, learning_rate, self.epsilon, &predictions);
            // Early stopping
            if do_early_stopping(avg_loss, prev_loss, &mut count_no_improve, params.early_stop_threshold(), params.patience(), params.early_stop_delta()) {
                println!("breaking at epoch {} with avg_loss {} and prev_loss {}", e, avg_loss, prev_loss);
                break;
            }
            prev_loss = avg_loss;
        }

        // Evaluate the test set after training, if it exists
        if self.test.len()!=0{
            let predictions = self.test();
            let mut loss_epoch = Tensor::from(0.0);
            let mut test_loss = vec![0.0; self.test.len()];
            for i in 0..self.test.len() {
                let loss_i = match params.loss() {
                    Loss::MAE => self.test[i].circuit_probability().l1_loss(&self.test.expected(i), Reduction::Mean),
                    Loss::MSE => self.test[i].circuit_probability().mse_loss(&self.test.expected(i), Reduction::Mean),
                };
                test_loss[i] = loss_i.to_f64();
                loss_epoch += loss_i;
            }
            self.log.log_test(&test_loss, self.epsilon, &predictions);
        }
    }
}

// --- Indexing the graph with dac indexes --- //
impl <const S: bool> std::ops::Index<DacIndex> for TensorLearner<S> 
{
    type Output = Dac<Tensor>;

    fn index(&self, index: DacIndex) -> &Self::Output {
        &self.train[index.0]
    }
}

