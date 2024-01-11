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
use rustc_hash::FxHashSet;
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

/// Abstraction used as a typesafe way of retrieving a `DAC` in the `Learner` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DacIndex(pub usize);

pub struct Learner<const S: bool>
{
    dacs: Vec<Dac<Float>>,
    test_dacs: Vec<Dac<Float>>,
    unsoftmaxed_distributions: Vec<Vec<f64>>,
    gradients: Vec<Vec<Float>>,
    lr: f64,
    expected_outputs: Vec<f64>,
    test_expected_outputs: Vec<f64>,
    log: Logger<S>,
    test_log: Logger<S>,
    epsilon: f64,
    learned_distributions: Vec<bool>,
}

fn loss_and_grad(loss:Loss, predictions: &Vec<f64>, expected: &Vec<f64>, mut dac_loss: Vec<f64>, mut dac_grad: Vec<f64>, _epsilon:f64) -> (Vec<f64>, Vec<f64>) {
    for i in 0..predictions.len() {
        match loss {
            Loss::MSE => {
                dac_loss[i] = (predictions[i] - expected[i]).powi(2);
                dac_grad[i] = 2.0 * (predictions[i] - expected[i]);
            },
            Loss::MAE => {
                dac_loss[i] = (predictions[i] - expected[i]).abs();
                dac_grad[i] = if predictions[i] > expected[i] {1.0} else if predictions[i]==expected[i] {0.0} else {-1.0};
            },
            /* Loss::Approx_MAE => {
                if predictions[i] >= expected[i]/(1.0+epsilon) && predictions[i] <= expected[i]*(1.0+epsilon) {
                    dac_loss[i] = 0.0;
                    dac_grad[i] = 0.0;
                }
                // TODO: loss with the nearest border or the expected?
                else {
                    dac_loss[i] = (predictions[i] - expected[i]).abs();
                    dac_grad[i] = if predictions[i] > expected[i] {1.0} else if predictions[i]==expected[i] {0.0} else {-1.0};
                }
            } */
        }
    }
    (dac_loss, dac_grad)
}

impl <const S: bool> Learner<S>
{
    /// Creates a new learner for the inputs given. Each inputs represent a query that needs to be
    /// solved, and the expected_outputs contains, for each query, its expected probability.
    pub fn new(inputs: Vec<PathBuf>, expected_outputs:Vec<f64>, epsilon:f64, branching: Branching, outfolder: Option<PathBuf>, jobs:usize, test_inputs:Vec<PathBuf>, test_expected:Vec<f64>) -> Self {
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
        let dacs = inputs.par_iter().map(|input| {
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
        let test_dacs = test_inputs.par_iter().map(|input| {
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
        let mut learner = Self { 
            dacs: vec![], 
            test_dacs: vec![],
            unsoftmaxed_distributions, 
            gradients: grads,
            lr: 0.0,
            expected_outputs: vec![],
            test_expected_outputs: vec![],
            log: Logger::default(),
            test_log: Logger::default(),
            epsilon,
            learned_distributions,
        };
        let mut distri_set: FxHashSet<usize> = FxHashSet::default();
        for (i, dac) in dacs.into_iter().enumerate() {
            if let Some(dac) = dac {
                for node in dac.iter() {
                    if let TypeNode::Distribution { d, .. } = dac[node].get_type() {
                        if learner.learned_distributions[d] {
                            distri_set.insert(d+1);
                        }
                    }
                }
                learner.dacs.push(dac);
                learner.expected_outputs.push(expected_outputs[i]);
            }
        }
        for (i, dac) in test_dacs.into_iter().enumerate() {
            if let Some(dac) = dac {
                learner.test_dacs.push(dac);
                learner.test_expected_outputs.push(test_expected[i]);
            }
        }
        //println!("learned distribution in queries {:?}", distri_set);
        learner.log = Logger::new(outfolder.as_ref(), learner.dacs.len(), true);
        if learner.test_dacs.len()!=0{ learner.test_log = Logger::new(outfolder.as_ref(), learner.test_dacs.len(), false);}
        learner
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

    pub fn get_expected_outputs(&self) -> &Vec<f64> {
        &self.expected_outputs
    }

    pub fn get_current_distributions(&self) -> &Vec<Vec<f64>> {
        &self.unsoftmaxed_distributions
    }

    pub fn get_number_dacs(&self) -> usize {
        self.dacs.len()
    }

    pub fn get_dac_i(&self, i: usize) -> &Dac<Float> {
        &self.dacs[i]
    }

    pub fn start_logger(&mut self) {
        self.log.start();
    }

    pub fn log_epoch(&mut self, loss:&Vec<f64>, lr:f64, predictions: &Vec<f64>) {
        self.log.log_epoch(loss, lr, self.epsilon, predictions);
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
        for dac in self.dacs.iter_mut() {
            dac.reset_distributions(&softmaxed);
        }
        self.dacs.par_iter_mut().for_each(|d| {
            d.evaluate();
            //println!("dac\n{}", d.as_graphviz());
        });
        self.dacs.iter().map(|d| d.get_circuit_probability().to_f64()).collect()
    }

    // Evaluate the different test DACs and return the results
    pub fn test(&mut self) -> Vec<f64> {
        let softmaxed = self.get_softmaxed_array();
        for dac in self.test_dacs.iter_mut() {
            dac.reset_distributions(&softmaxed);
        }
        self.dacs.par_iter_mut().for_each(|d| {
            d.evaluate();
            //println!("dac\n{}", d.as_graphviz());
        });
        self.dacs.iter().map(|d| d.get_circuit_probability().to_f64()).collect()
    }

    // --- Gradient computation --- //

    // Compute the gradient of the distributions, from the different DAC queries
    pub fn compute_gradients(&mut self, gradient_loss: &Vec<f64>) {
        self.zero_grads();
        for dac_id in 0..self.dacs.len() {
            self.dacs[dac_id].zero_paths();
            let len = self.dacs[dac_id].nodes.len();
            self.dacs[dac_id].nodes[len-1].set_path_value(f128!(1.0));
            // Iterate on all nodes from the DAC, top-down way
            for node in self.dacs[dac_id].iter_rev() {
                let start = self.dacs[dac_id][node].get_input_start();
                let end = start + self.dacs[dac_id][node].get_number_inputs();
                let value = self.dacs[dac_id][node].get_value().to_f64();
                let path_val = self.dacs[dac_id][node].get_path_value();

                // Update the path value for the children sum, product nodes 
                // and compute the gradient for the children leaf distributions
                for child_index in start..end {
                    let child = self.dacs[dac_id].get_input_at(child_index);
                    match self.dacs[dac_id][node].get_type() {
                        TypeNode::Product => {
                            let mut val = f128!(0.0);
                            if self.dacs[dac_id][child].get_value().to_f64() != 0.0 {
                                val = path_val.clone() * &value / self.dacs[dac_id][child].get_value().to_f64();
                            }
                            self.dacs[dac_id][child].add_to_path_value(val)
                        },
                        TypeNode::Sum => {
                            self.dacs[dac_id][child].add_to_path_value(path_val.clone());
                        },
                        TypeNode::Partial => { },
                        TypeNode::Distribution { .. } => {},
                    }
                    if let TypeNode::Distribution { d, v } = self.dacs[dac_id][child].get_type() {
                        // Compute the gradient for children that are leaf distributions
                        let mut factor = path_val.clone() * gradient_loss[dac_id];
                        if let TypeNode::Product = self.dacs[dac_id][node].get_type() {
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

    pub fn update_distributions(&mut self) {
        for (i, (distribution, grad)) in self.unsoftmaxed_distributions.iter_mut().zip(self.gradients.iter()).enumerate() {
            if self.learned_distributions[i]{
                for (value, grad) in distribution.iter_mut().zip(grad.iter()) {
                    *value -= (self.lr * grad.clone()).to_f64();
                }
            }
        }
    }
}

impl<const S: bool> Learning for Learner<S> {

    fn train(&mut self, nepochs:usize, init_lr:f64, loss: Loss, timeout:i64,) {
        self.lr = init_lr;
        let lr_drop: f64 = 0.75;
        let epoch_drop = 100.0;
        let stopping_criterion = 0.0001;
        let mut prev_loss = 1.0;
        let delta_early_stop = 0.00001;
        let mut count_no_improve = 0;
        let patience = 5;
        self.log.start();
        let start = chrono::Local::now();

        let mut dac_loss = vec![0.0; self.dacs.len()];
        let mut dac_grad = vec![0.0; self.dacs.len()];

        for e in 0..nepochs {
            if (chrono::Local::now() - start).num_seconds() > timeout { break;}
            let do_print = e % 500 == 0;
            self.lr = init_lr * lr_drop.powf(((1+e) as f64/ epoch_drop).floor());
            if do_print{println!("Epoch {} lr {}", e, self.lr);}
            let predictions = self.evaluate();
            // TODO: Maybe we can pass that in the logger and do something like self.log.print()
            if do_print { println!("--- Epoch {} ---\n Predictions: {:?} \nExpected: {:?}\n", e, predictions, self.expected_outputs);}
            (dac_loss, dac_grad) = loss_and_grad(loss, &predictions, &self.expected_outputs, dac_loss, dac_grad, self.epsilon);
            self.compute_gradients(&dac_grad);
            //if do_print{ println!("Gradients: {:?}", self.gradients);}
            self.update_distributions();
            self.log.log_epoch(&dac_loss, self.lr, self.epsilon, &predictions);
            let avg_loss = dac_loss.iter().sum::<f64>() / dac_loss.len() as f64;
            if do_print { println!("Loss: {}", avg_loss);}
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
            dac_loss.fill(0.0);
            dac_grad.fill(0.0);
        }

        if self.test_dacs.len()!=0{ 
            self.test_log.start();
            let predictions = self.test();
            let (dac_loss, _) = loss_and_grad(loss, &predictions, &self.test_expected_outputs, dac_loss, dac_grad, self.epsilon);
            self.test_log.log_epoch(&dac_loss, self.lr, self.epsilon, &predictions);
        }
    }
}

// --- Indexing the graph with dac indexes --- //
impl <const S: bool> std::ops::Index<DacIndex> for Learner<S> 
{
    type Output = Dac<Float>;

    fn index(&self, index: DacIndex) -> &Self::Output {
        &self.dacs[index.0]
    }
}

