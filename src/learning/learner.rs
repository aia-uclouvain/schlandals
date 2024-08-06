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
//! Note that the 'tensor_learner' module is equivalent to this learner but uses tensor automatic
//! backpropagation while this learner computes the gradients for floats.

use std::path::PathBuf;
use std::time::{Instant, Duration};
use crate::ac::ac::*;
use crate::ac::node::NodeType;
use super::logger::Logger;
use crate::{parser::*, ApproximateMethod};
use crate::Branching;
use rayon::prelude::*;
use super::Learning;
use crate::common::F128;
use super::utils::*;
use super::*;
use rug::{Assign, Float};

/// Abstraction used as a typesafe way of retrieving a `DAC` in the `Learner` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DacIndex(pub usize);

/// Structure used to learn the distribution parameters from a set of queries
pub struct Learner<const S: bool> {
    train: Dataset<Float>,
    test: Dataset<Float>,
    unsoftmaxed_distributions: Vec<Vec<f64>>,
    gradients: Vec<Vec<Float>>,
    log: Logger<S>,
    learned_distributions: Vec<bool>,
    epsilon: f64,
}

impl <const S: bool> Learner<S> {
    /// Creates a new learner for the given inputs. Each inputs represent a query that needs to be
    /// solved, and the expected_outputs contains, for each query, its expected probability.
    pub fn new(inputs: &Vec<PathBuf>, mut expected_outputs:Vec<f64>, epsilon:f64, approx:ApproximateMethod, branching: Branching, outfolder: Option<PathBuf>, 
               jobs:usize, compile_timeout: u64, test_inputs:Vec<PathBuf>, mut expected_test: Vec<f64>, equal_init: bool) -> Self {
        
        rayon::ThreadPoolBuilder::new().num_threads(jobs).build_global().unwrap();
        
        // Retrieves the distributions values and computes their unsoftmaxed values
        // and initializes the gradients to 0
        let distributions = distributions_from_cnf(&inputs[0]);
        let mut grads: Vec<Vec<Float>> = vec![];
        let mut unsoftmaxed_distributions: Vec<Vec<f64>> = vec![];
        let mut eps = epsilon;
        for distribution in distributions.iter() {
            if !equal_init {
                let unsoftmaxed_vector = distribution.iter().map(|p| p.log(std::f64::consts::E)).collect::<Vec<f64>>();
                unsoftmaxed_distributions.push(unsoftmaxed_vector);
            }
            else {
                let unsoftmaxed_vector = distribution.iter().map(|_| (1.0/(distribution.len() as f64)).log(std::f64::consts::E)).collect::<Vec<f64>>();
                unsoftmaxed_distributions.push(unsoftmaxed_vector);
            }
            grads.push(vec![F128!(0.0); distribution.len()]);
        }
        // Retrieves which distributions are learned
        let learned_distributions = learned_distributions_from_cnf(&inputs[0]);

        // Compiling the train and test queries into arithmetic circuits
        let mut train_dacs = generate_dacs(inputs, branching, epsilon, approx, compile_timeout);
        if approx == ApproximateMethod::LDS {
            eps = 0.0;
            let mut present_distributions = vec![0; distributions.len()];
            let mut cnt_unfinished = 0;
            for dac in train_dacs.iter() {
                if !dac.is_complete() {
                    cnt_unfinished += 1;
                }
                for node in dac.iter() {
                    let start = dac[node].input_start();
                    let end = start + dac[node].number_inputs();
                    for i in start..end {
                        if let NodeType::Distribution { d, .. } = dac[dac.input_at(i)].get_type() {
                            present_distributions[d] += 1;
                        }
                    }
                }
                eps += dac.epsilon();
            }
            println!("Present distributions counted {}/{}", present_distributions.iter().filter(|b| **b!=0).count(), present_distributions.len());
            println!("Occurance of distributions {:?}", present_distributions);//.iter().filter(|b| **b!=0).collect::<Vec<&usize>>());
            println!("Unfinished DACs: {}, total {}", cnt_unfinished, train_dacs.len());
            println!("Epsilon: {}", eps);
        }
        let mut test_dacs = generate_dacs(&test_inputs, branching, epsilon, ApproximateMethod::Bounds, u64::MAX);
        // Creating train and test datasets
        let mut train_data = vec![];
        let mut train_expected = vec![];
        let mut test_data = vec![];
        let mut test_expected = vec![];
        while let Some(d) = train_dacs.pop() {
            let expected = expected_outputs.pop().unwrap();
            train_data.push(d);
            train_expected.push(F128!(expected));
        }
        while let Some(d) = test_dacs.pop() {
            let expected = expected_test.pop().unwrap();
            test_data.push(d);
            test_expected.push(F128!(expected));
        }
        let train_dataset = Dataset::new(train_data, train_expected);
        let test_dataset = Dataset::new(test_data, test_expected);
        // Initializing the logger
        let log = Logger::new(outfolder.as_ref(), train_dataset.len(), test_dataset.len());
        
        Self { 
            train: train_dataset,
            test: test_dataset,
            unsoftmaxed_distributions, 
            gradients: grads,
            log,
            learned_distributions,
            epsilon: eps,
        }
    }

    // --- Getters --- //

    /// Return the softmax values for the paramater of the given distribution
    fn get_softmaxed(&self, distribution: usize) -> Vec<Float> {
        softmax(&self.unsoftmaxed_distributions[distribution])
    }

    /// Return the probability of the given value for the given distribution
    pub fn get_probability(&self, distribution: usize, index: usize) -> f64 {
        self.get_softmaxed(distribution)[index].to_f64()
    }

    /// Returns a double vector of tensors. Each entry (d,i) is a tensor representing the softmaxed
    /// version of the i-th value of vector d
    pub fn get_softmaxed_array(&self) -> Vec<Vec<Float>> {
        let mut softmaxed: Vec<Vec<Float>> = vec![];
        for distribution in self.unsoftmaxed_distributions.iter() {
            softmaxed.push(softmax(distribution));
        }
        softmaxed
    }

    // --- Setters --- //

    /// Set the gradients of the parameters to 0
    pub fn zero_grads(&mut self) {
        for grad in self.gradients.iter_mut() {
            for el in grad.iter_mut() {
                el.assign(0.0);
            }
        }
    }

    // --- Evaluation --- //

    // Evaluate the different train DACs and return the results
    pub fn evaluate(&mut self) -> Vec<f64> {
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
    pub fn test(&mut self) -> Vec<f64> {
        let softmaxed = self.get_softmaxed_array();
        for dac in self.test.get_queries_mut() {
            dac.reset_distributions(&softmaxed);
        }
        self.test.get_queries_mut().par_iter_mut().for_each(|d| {
            d.evaluate();
        });
        self.test.get_queries().iter().map(|d| d.circuit_probability().to_f64()).collect()
    }

    fn recompile_dacs(&mut self, inputs: &Vec<PathBuf>, branching: Branching, approx:ApproximateMethod, compile_timeout: u64) {
        let mut train_dacs = generate_dacs(inputs, branching, self.epsilon, approx, compile_timeout);
        let mut train_data = vec![];
        let mut eps = 0.0;
        while let Some(d) = train_dacs.pop() {
            eps += d.epsilon();
            train_data.push(d);
        }
        self.train.set_queries(train_data);
        self.epsilon = eps;
    }

    // --- Gradient computation --- //

    // Compute the gradient of the distributions, from the different DAC queries
    // The computation is done in a top-down way, starting from the root node
    // and uses the chaine rule of the derivative to cumulatively compute the gradient in the leaves
    pub fn compute_gradients(&mut self, gradient_loss: &[f64]) {
        self.zero_grads();
        for query_id in 0..self.train.len() {
            self.train[query_id].zero_paths();
            // Iterate on all nodes from the DAC, top-down way
            for node in self.train[query_id].iter_rev() {

                let start = self.train[query_id][node].input_start();
                let end = start + self.train[query_id][node].number_inputs();
                let value = self.train[query_id][node].value().to_f64();
                let path_val = self.train[query_id][node].path_value();

                // Update the path value for the children sum, product nodes 
                // and compute the gradient for the children leaf distributions
                for child_index in start..end {
                    let child = self.train[query_id].input_at(child_index);
                    match self.train[query_id][node].get_type() {
                        NodeType::Product => {
                            // If it is a product node, we need to divide the path value by the value of the child
                            // This is equivalent to multiplying the values of the other children
                            // If the value of the child is 0, then the path value is simply 0
                            let mut val = F128!(0.0);
                            if self.train[query_id][child].value().to_f64() != 0.0 {
                                val = path_val.clone() * &value / self.train[query_id][child].value().to_f64();
                            }
                            self.train[query_id][child].add_to_path_value(val);
                        },
                        NodeType::Sum => {
                            // If it is a sum node, we simply propagate the path value to the children
                            self.train[query_id][child].add_to_path_value(path_val.clone());
                        },
                        NodeType::Distribution { .. } => {},
                    }
                    if let NodeType::Distribution { d, v } = self.train[query_id][child].get_type() {
                        // Compute the gradient for children that are leaf distributions
                        let mut factor = path_val.clone() * gradient_loss[query_id];
                        if self.train[query_id][node].is_product() {
                            factor *= value;
                            factor /= self.get_probability(d, v);
                        }
                        // Compute the gradient contribution for the value used in the node 
                        // and all the other possible values of the distribution (derivative of the softmax)
                        let mut sum_other_w = F128!(0.0);
                        let child_w = self.get_probability(d, v);
                        for params in (0..self.unsoftmaxed_distributions[d].len()).filter(|i| *i != v) {
                            let weight = self.get_probability(d, params);
                            self.gradients[d][params] -= factor.clone() * weight * child_w;
                            sum_other_w += weight;
                        }
                        self.gradients[d][v] += factor * child_w * sum_other_w;
                    }
                }
            }
        }
    }

    /// Update the distributions with the computed gradients and the learning rate, following an SGD approach
    pub fn update_distributions(&mut self, learning_rate: f64) {
        for (i, (distribution, grad)) in self.unsoftmaxed_distributions.iter_mut().zip(self.gradients.iter()).enumerate() {
            if self.learned_distributions[i]{
                for (value, grad) in distribution.iter_mut().zip(grad.iter()) {
                    *value -= (learning_rate * grad.clone()).to_f64();
                }
            }
        }
    }
}

impl<const S: bool> Learning for Learner<S> {
    /// Training loop for the train dacs, using the given training parameters
    fn train(&mut self, params: &LearnParameters, inputs: &Vec<PathBuf>, branching: Branching, approx:ApproximateMethod, compile_timeout: u64) {
        let mut prev_loss = 1.0;
        let mut count_no_improve = 0;
        self.log.start();
        let start = Instant::now();
        let to = Duration::from_secs(params.learning_timeout());
        
        // Evaluate the test set before training, if it exists
        if self.test.len() != 0 {
            let predictions = self.test();
            let test_loss = predictions.iter().copied().enumerate().map(|(i, prediction)| params.loss().loss(prediction, self.test.expected(i).to_f64())).collect::<Vec<f64>>();
            self.log.log_test(&test_loss, self.epsilon, &predictions);
        }

        // Training loop
        let mut train_loss = vec![0.0; self.train.len()];
        let mut train_grad = vec![0.0; self.train.len()];
        for e in 0..params.nepochs() {
            // Update the learning rate
            let learning_rate = params.lr() * params.lr_drop().powf(((1+e) as f64/ params.epoch_drop() as f64).floor());
            // Forward pass
            let predictions = self.evaluate();
            // Compute the loss and the gradients
            for i in 0..predictions.len() {
                train_loss[i] = params.loss().loss(predictions[i], self.train.expected(i).to_f64());
                train_grad[i] = params.loss().gradient(predictions[i], self.train.expected(i).to_f64());
                if params.e_weighted() && self.epsilon != 0.0 {
                    //train_loss[i] *= 1.0 - self.train[i].epsilon()/self.epsilon;
                    let l = self.train.len() as f64;
                    train_grad[i] *= (1.0 - self.train[i].epsilon()/self.epsilon) * l / (l - 1.0);
                }
            }
            let avg_loss = train_loss.iter().sum::<f64>() / train_loss.len() as f64;
            self.compute_gradients(&train_grad);
            // Update the parameters
            self.update_distributions(learning_rate);
            // Log the epoch
            self.log.log_epoch(&train_loss, learning_rate, self.epsilon, &predictions);
            // Early stopping
            if do_early_stopping(avg_loss, prev_loss, &mut count_no_improve, params.early_stop_threshold, params.patience(), params.early_stop_delta()) {
                println!("breaking at epoch {} with avg_loss {} and prev_loss {}", e, avg_loss, prev_loss);
                break;
            }
            prev_loss = avg_loss;

            // Timeout
            if start.elapsed() > to {
                break;
            }

            if e != params.nepochs() - 1 && params.recompile() { //&& e!=0 && e%10==0 {
                println!("Recompiling at epoch {}", e);
                self.recompile_dacs(inputs, branching, approx, compile_timeout);
            }

            // TODO: Add a verbosity command line argument
            /*
            let do_print = e % 500 == 0;
            if do_print{println!("Epoch {} lr {}", e, self.lr);}
            if do_print { println!("--- Epoch {} ---\n Predictions: {:?} \nExpected: {:?}\n", e, predictions, self.expected_outputs);}
            //if do_print{ println!("Gradients: {:?}", self.gradients);}
            if do_print { println!("Loss: {}", avg_loss);}
            */
        }
        
        // Evaluate the test set after training, if it exists
        if self.test.len() != 0 {
            let predictions = self.test();
            let test_loss = predictions.iter().copied().enumerate().map(|(i, prediction)| params.loss().loss(prediction, self.test.expected(i).to_f64())).collect::<Vec<f64>>();
            self.log.log_test(&test_loss, self.epsilon, &predictions);
        }
    }
}

// --- Indexing the graph with dac indexes --- //
impl <const S: bool> std::ops::Index<DacIndex> for Learner<S> 
{
    type Output = Dac<Float>;

    fn index(&self, index: DacIndex) -> &Self::Output {
        &self.train[index.0]
    }
}
