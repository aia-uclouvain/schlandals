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

use std::path::PathBuf;

use rug::{Assign, Float};
use crate::diagrams::dac::dac::*;
use crate::diagrams::dac::node::TypeNode;
use super::logger::Logger;
use search_trail::StateManager;
use crate::branching::*;
use crate::parser::*;
use crate::propagator::Propagator;
use crate::core::components::ComponentExtractor;
use crate::Branching;
use crate::Loss;
use crate::solvers::DACCompiler;
use crate::solvers::*;
use rayon::prelude::*;
use super::Learning;
use crate::common::f128;
use super::utils::*;
use super::*;
use std::time::{Instant, Duration};

/// Abstraction used as a typesafe way of retrieving a `DAC` in the `Learner` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DacIndex(pub usize);

pub struct Learner<const S: bool>
{
    train: Dataset<Float>,
    test: Dataset<Float>,
    unsoftmaxed_distributions: Vec<Vec<f64>>,
    gradients: Vec<Vec<Float>>,
    log: Logger<S>,
    learned_distributions: Vec<bool>,
}

impl <const S: bool> Learner<S>
{
    /// Creates a new learner for the inputs given. Each inputs represent a query that needs to be
    /// solved, and the expected_outputs contains, for each query, its expected probability.
    pub fn new(inputs: Vec<PathBuf>, mut expected_outputs:Vec<f64>, epsilon:f64, branching: Branching, outfolder: Option<PathBuf>, jobs:usize, test_inputs:Vec<PathBuf>, mut expected_test: Vec<f64>) -> Self {
        rayon::ThreadPoolBuilder::new().num_threads(jobs).build_global().unwrap();
        let distributions = distributions_from_cnf(&inputs[0]);
        let mut grads: Vec<Vec<Float>> = vec![];
        let mut unsoftmaxed_distributions: Vec<Vec<f64>> = vec![];
        for distribution in distributions.iter() {
            let unsoftmaxed_vector = distribution.iter().map(|p| p.log(std::f64::consts::E)).collect::<Vec<f64>>();
            unsoftmaxed_distributions.push(unsoftmaxed_vector);
            grads.push(vec![f128!(0.0); distribution.len()]);
        }
        let learned_distributions = learned_distributions_from_cnf(&inputs[0]);

        // Compiling the train and test queries into arithmetic circuits
        let mut train_dacs = inputs.par_iter().map(|input| {
            // We compile the input. This can either be a .cnf file or a fdac file.
            // If the file is a fdac file, then we read directly from it
            match type_of_input(input) {
                FileType::CNF => {
                    println!("Compiling {}", input.to_str().unwrap());
                    // The input is a CNF file, we need to compile it from scratch
                    // First, we need to know how much distributions are needed to compute the
                    // query.
                    let mut compiler = make_compiler!(input, branching, epsilon);
                    if let Some(mut dac) = compile!(compiler) {
                        dac.optimize_structure();
                        //println!("nb approx {}, nb nodes {}", dac.iter().filter(|n| dac[*n].is_partial()).count(), dac.nodes.len());
                        Some(dac)
                    } else {
                        None
                    }
                },
                FileType::FDAC => {
                    println!("Reading {}", input.to_str().unwrap());
                    // The query has already been compiled, we just read from the file.
                    Some(Dac::from_file(input))
                }
            }
        }).collect::<Vec<_>>();

        let mut test_dacs = test_inputs.par_iter().map(|input| {
            // We compile the input. This can either be a .cnf file or a fdac file.
            // If the file is a fdac file, then we read directly from it
            match type_of_input(input) {
                FileType::CNF => {
                    println!("Compiling {}", input.to_str().unwrap());
                    // The input is a CNF file, we need to compile it from scratch
                    // First, we need to know how much distributions are needed to compute the
                    // query.
                    let mut compiler = make_compiler!(input, branching, epsilon);
                    compile!(compiler)
                },
                FileType::FDAC => {
                    println!("Reading {}", input.to_str().unwrap());
                    // The query has already been compiled, we just read from the file.
                    Some(Dac::from_file(input))
                }
            }
        }).collect::<Vec<_>>();

        let mut train_data = vec![];
        let mut train_expected = vec![];
        let mut test_data = vec![];
        let mut test_expected = vec![];

        while !train_dacs.is_empty() {
            let d = train_dacs.pop().unwrap();
            let expected = expected_outputs.pop().unwrap();
            if let Some(dac) = d {
                train_data.push(dac);
                train_expected.push(expected);
            }
        }

        while !test_dacs.is_empty() {
            let d = test_dacs.pop().unwrap();
            let expected = expected_test.pop().unwrap();
            if let Some(dac) = d {
                test_data.push(dac);
                test_expected.push(expected);
            }
        }
        let train_dataset = Dataset::new(train_data, train_expected);
        let test_dataset = Dataset::new(test_data, test_expected);

        let log = Logger::new(outfolder.as_ref(), train_dataset.len(), true);
        Self { 
            train: train_dataset,
            test: test_dataset,
            unsoftmaxed_distributions, 
            gradients: grads,
            log,
            learned_distributions,
        }
    }

    // --- Getters --- //
    fn get_softmaxed(&self, distribution: usize) -> Vec<Float> {
        softmax(&self.unsoftmaxed_distributions[distribution])
    }

    pub fn get_probability(&self, distribution: usize, index: usize) -> f64 {
        self.get_softmaxed(distribution)[index].to_f64()
    }

    pub fn get_softmaxed_array(&self) -> Vec<Vec<Float>> {
        let mut softmaxed: Vec<Vec<Float>> = vec![];
        for distribution in self.unsoftmaxed_distributions.iter() {
            softmaxed.push(softmax(distribution));
        }
        softmaxed
    }

    pub fn get_current_distributions(&self) -> &Vec<Vec<f64>> {
        &self.unsoftmaxed_distributions
    }

    // --- Setters --- //
    pub fn zero_grads(&mut self) {
        for grad in self.gradients.iter_mut() {
            for el in grad.iter_mut() {
                el.assign(0.0);
            }
        }
    }


    // --- Evaluation --- //

    // Evaluate the different DACs and return the results
    pub fn evaluate(&mut self) -> Vec<f64> {
        let softmaxed = self.get_softmaxed_array();
        for dac in self.train.get_queries_mut() {
            dac.reset_distributions(&softmaxed);
        }
        self.train.get_queries_mut().par_iter_mut().for_each(|d| {
            d.evaluate();
        });
        self.train.get_queries().iter().map(|d| d.get_circuit_probability().to_f64()).collect()
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
        self.test.get_queries().iter().map(|d| d.get_circuit_probability().to_f64()).collect()
    }

    // --- Gradient computation --- //

    // Compute the gradient of the distributions, from the different DAC queries
    pub fn compute_gradients(&mut self, gradient_loss: &Vec<f64>) {
        self.zero_grads();
        for query_id in 0..self.train.len() {
            self.train[query_id].zero_paths();
            // Iterate on all nodes from the DAC, top-down way
            for node in self.train[query_id].iter_rev() {

                let start = self.train[query_id][node].get_input_start();
                let end = start + self.train[query_id][node].get_number_inputs();
                let value = self.train[query_id][node].get_value().to_f64();
                let path_val = self.train[query_id][node].get_path_value();

                // Update the path value for the children sum, product nodes 
                // and compute the gradient for the children leaf distributions
                for child_index in start..end {
                    let child = self.train[query_id].get_input_at(child_index);
                    match self.train[query_id][node].get_type() {
                        TypeNode::Product => {
                            let mut val = f128!(0.0);
                            if self.train[query_id][child].get_value().to_f64() != 0.0 {
                                val = path_val.clone() * &value / self.train[query_id][child].get_value().to_f64();
                            }
                            self.train[query_id][child].add_to_path_value(val);
                        },
                        TypeNode::Sum => {
                            self.train[query_id][child].add_to_path_value(path_val.clone());
                        },
                        TypeNode::Partial => { },
                        TypeNode::Distribution { .. } => {},
                    }
                    if let TypeNode::Distribution { d, v } = self.train[query_id][child].get_type() {
                        // Compute the gradient for children that are leaf distributions
                        let mut factor = path_val.clone() * gradient_loss[query_id];
                        if self.train[query_id][node].is_product() {
                            factor *= value;
                            factor /= self.get_probability(d, v);
                        }
                        // Compute the gradient contribution for the value used in the node and all the other possible values of the distribution
                        let mut sum_other_w = f128!(0.0);
                        let child_w = self.get_probability(d, v);
                        for params in (0..self.unsoftmaxed_distributions[d].len()).filter(|i| *i != v) {
                            let weight = self.get_probability(d, params);
                            self.gradients[d][params] -= factor.clone() * weight.clone() * child_w.clone();
                            sum_other_w += weight.clone();
                        }
                        self.gradients[d][v] += factor * child_w * sum_other_w;
                    }
                }
            }
        }
    }

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

    fn train(&mut self, nepochs:usize, init_lr:f64, loss: Loss, timeout: u64) {
        let lr_drop: f64 = 0.75;
        let epoch_drop = 100.0;
        let stopping_criterion = 0.0001;
        let mut prev_loss = 1.0;
        let delta_early_stop = 0.00001;
        let mut count_no_improve = 0;
        let patience = 5;
        self.log.start();
        let start = Instant::now();
        let to = Duration::from_secs(timeout);

        let mut train_loss = vec![0.0; self.train.len()];
        let mut train_grad = vec![0.0; self.train.len()];

        for e in 0..nepochs {
            if start.elapsed() > to {
                break;
            }
            let learning_rate = init_lr * lr_drop.powf(((1+e) as f64/ epoch_drop).floor());
            let predictions = self.evaluate();
            for i in 0..predictions.len() {
                train_loss[i] = loss.loss(predictions[i], self.train.expected(i));
                train_grad[i] = loss.gradient(predictions[i], self.train.expected(i));
            }
            self.compute_gradients(&train_grad);
            self.update_distributions(learning_rate);
            let avg_loss = train_loss.iter().sum::<f64>() / train_loss.len() as f64;
            if (avg_loss-prev_loss).abs()<delta_early_stop {
                count_no_improve += 1;
            }
            else {
                count_no_improve = 0;
            }
            if (avg_loss < stopping_criterion) || count_no_improve>=patience {
                println!("breaking at epoch {} with avg_loss {} and prev_loss {}", e, avg_loss, prev_loss);
                break;
            }
            prev_loss = avg_loss;

            // TODO: Add a verbosity command line argument
            /*
            let do_print = e % 500 == 0;
            if do_print{println!("Epoch {} lr {}", e, self.lr);}
            if do_print { println!("--- Epoch {} ---\n Predictions: {:?} \nExpected: {:?}\n", e, predictions, self.expected_outputs);}
            //if do_print{ println!("Gradients: {:?}", self.gradients);}
            self.update_distributions();
            self.log.log_epoch(&dac_loss, self.lr, self.epsilon, &predictions);
            if do_print { println!("Loss: {}", avg_loss);}
            */
        }

        if self.test.len() != 0 {
            let predictions = self.test();
            let test_loss = predictions.iter().copied().enumerate().map(|(i, prediction)| loss.loss(prediction, self.test.expected(i))).collect::<Vec<f64>>();
            let average_loss = test_loss.iter().sum::<f64>() / test_loss.len() as f64;
            println!("Average test loss : {}", average_loss);
            // TODO: Log somewhere the loss of the test set. Maybe we can add this small loop also
            // at the beginning of the method. Should we output on stdout ?

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

