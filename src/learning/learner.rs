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

use std::fmt;
use std::path::PathBuf;
use std::fs::File;

use std::io::{self, Write};
use rug::{Assign, Float};
use crate::common::*;
use crate::diagrams::dac::dac::*;
use crate::diagrams::dac::node::TypeNode;
use super::logger::Logger;
use search_trail::StateManager;
use crate::branching::*;
use crate::parser::*;
use crate::propagator::Propagator;
use crate::preprocess::Preprocessor;
use crate::core::components::ComponentExtractor;
use crate::Branching;
use crate::Loss;
use crate::solvers::DACCompiler;
use crate::solvers::*;

/// Abstraction used as a typesafe way of retrieving a `DAC` in the `Learner` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DacIndex(pub usize);
use rayon::prelude::*;

pub struct Learner<const S: bool>
{
    dacs: Vec<Dac>,
    unsoftmaxed_distributions: Vec<Vec<f64>>,
    gradients: Vec<Vec<Float>>,
    is_distribution_learned: Vec<bool>,
    lr: f64,
    expected_distribution: Vec<Vec<f64>>,
    expected_outputs: Vec<f64>,
    log: Logger<S>,
    outfolder: Option<PathBuf>,
    epsilon: f64,
    ratio_learn: f64,
}

/// Calculates the softmax (the normalized exponential) function, which is a generalization of the
/// logistic function to multiple dimensions.
///
/// Takes in a vector of real numbers and normalizes it to a probability distribution such that
/// each of the components are in the interval (0, 1) and the components add up to 1. Larger input
/// components correspond to larger probabilities.
/// From https://docs.rs/compute/latest/src/compute/functions/statistical.rs.html#43-46
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let sum_exp: f64 = x.iter().map(|i| i.exp()).sum();
    x.iter().map(|i| i.exp() / sum_exp).collect()
}

fn loss_and_grad(loss:Loss, predictions: &Vec<f64>, expected: &Vec<f64>, mut dac_loss: Vec<f64>, mut dac_grad: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    for i in 0..predictions.len() {
        match loss {
            Loss::MSE => {
                dac_loss[i] = (predictions[i] - expected[i]).powi(2);
                dac_grad[i] = 2.0 * (predictions[i] - expected[i]);
            },
            Loss::MAE => {
                dac_loss[i] = (predictions[i] - expected[i]).abs();
                dac_grad[i] = if predictions[i] > expected[i] {1.0} else {-1.0};
            },
        }
    }
    (dac_loss, dac_grad)
}

impl <const S: bool> Learner<S>
{

    fn get_number_needed_distributions(input: &PathBuf, branching: Branching) -> usize {
        let mut state = StateManager::default();
        let mut propagator = Propagator::new(&mut state);
        let mut graph = graph_from_ppidimacs(&input, &mut state);
        let mut component_extractor = ComponentExtractor::new(&graph, &mut state);
        propagator.init(graph.number_clauses());
        let mut branching_heuristic: Box<dyn BranchingDecision> = match branching {
            Branching::MinInDegree => Box::<MinInDegree>::default(),
            Branching::MinOutDegree => Box::<MinOutDegree>::default(),
            Branching::MaxDegree => Box::<MaxDegree>::default(),
            Branching::VSIDS => Box::<VSIDS>::default(),
        };
        Preprocessor::new(&mut graph, &mut state, branching_heuristic.as_mut(), &mut propagator, &mut component_extractor).preprocess(false);
        graph.distributions_iter().filter(|d| graph[*d].is_constrained(&state)).count()
    }

    /// Creates a new learner for the inputs given. Each inputs represent a query that needs to be
    /// solved, and the expected_outputs contains, for each query, its expected probability.
    pub fn new(inputs: Vec<PathBuf>, expected_outputs:Vec<f64>, epsilon:f64, branching: Branching, outfolder: Option<PathBuf>, ratio_learn:f64) -> Self {
        let distributions = distributions_from_cnf(&inputs[0]);
        println!("distributions {:?}", distributions);
        // TODO what about fdist files ?
        let mut grads: Vec<Vec<Float>> = vec![];
        let mut unsoftmaxed_distributions: Vec<Vec<f64>> = vec![];
        let mut rand_init: Vec<Vec<f64>> = vec![];
        for distribution in distributions.iter() {
            // Computing a random initial value in case the distribution must be learned
            let random_probabilities = (0..distribution.len()).map(|_| rand::random::<f64>().log(std::f64::consts::E)).collect::<Vec<f64>>();
            let unsoftmaxed_vector = distribution.iter().map(|p| p.log(std::f64::consts::E)).collect::<Vec<f64>>();
            rand_init.push(random_probabilities);
            unsoftmaxed_distributions.push(unsoftmaxed_vector);
            grads.push(vec![f128!(0.0); distribution.len()]);
        }
        let dacs = inputs.par_iter().map(| input| {
            // We compile the input. This can either be a .cnf file or a fdac file.
            // If the file is a fdac file, then we read directly from it
            let mut d = match type_of_input(input) {
                FileType::CNF => {
                    println!("Compiling {}", input.to_str().unwrap());
                    // The input is a CNF file, we need to compile it from scratch
                    // First, we need to know how much distributions are needed to compute the
                    // query.
                    let number_distribution = Self::get_number_needed_distributions(input, branching);
                    // TODO: Maybe use a fixed threshold for easy instance
                    let learned = (ratio_learn * number_distribution as f64).ceil() as usize;
                    //println!("Number of distributions: {}, learned: {}", number_distribution, learned);
                    let compiler = make_compiler!(input, branching, learned);
                    compile!(compiler)
                },
                FileType::FDAC => {
                    println!("Reading {}", input.to_str().unwrap());
                    // The query has already been compiled, we just read from the file.
                    Some(Dac::from_file(input))
                }
            };
            // We handle the compiled circuit, if present.
            if let Some(ref mut dac) = d.as_mut() {
                if dac.has_cutoff_nodes() {
                    // The circuit has some nodes that have been cut-off. This means that, when
                    // evaluating the circuit, they need to be solved. Hence we stock a solver
                    // for this query.
                    let solver = make_solver!(input, branching, epsilon, None, false);
                    dac.set_solver(solver);
                }
            }
            d
        }).collect::<Vec<_>>();
        let logger = Logger::new(outfolder.as_ref(), dacs.iter().filter(|d| d.is_some()).count());
        let mut learner = Self { 
            dacs: vec![], 
            unsoftmaxed_distributions, 
            gradients: grads,
            is_distribution_learned: vec![true; distributions.len()],
            lr: 0.0,
            expected_distribution: distributions,
            expected_outputs: vec![],
            log: logger,
            outfolder,
            epsilon,
            ratio_learn,
        };

        for (i, dac) in dacs.into_iter().enumerate() {
            if let Some(dac) = dac {
                learner.dacs.push(dac);
                learner.expected_outputs.push(expected_outputs[i]);
            }
        }
        if ratio_learn < 1.0 {
            learner.is_distribution_learned = vec![false; learner.expected_distribution.len()];
            for dac_i in 0..learner.dacs.len() {
                for (d, _) in learner.dacs[dac_i].distribution_mapping.keys(){
                    learner.is_distribution_learned[d.0] = true;
                    learner.unsoftmaxed_distributions[d.0] = rand_init[d.0].clone();
                }
            }
        }
        else {
            learner.unsoftmaxed_distributions = rand_init;
        }
        // Send the randomized distributions to the solvers
        learner.update_distributions();

        println!("is_distrib_learned {:?}", learner.is_distribution_learned);
        learner.to_folder();
        learner
    }

    // --- Getters --- //
    fn get_softmaxed(&self, distribution: usize) -> Vec<f64> {
        softmax(&self.unsoftmaxed_distributions[distribution])
    }

    pub fn get_probability(&self, distribution: usize, index: usize) -> f64 {
        self.get_softmaxed(distribution)[index]
    }

    pub fn get_softmaxed_array(&self) -> Vec<Vec<f64>> {
        let mut softmaxed: Vec<Vec<f64>> = vec![];
        for distribution in self.unsoftmaxed_distributions.iter() {
            softmaxed.push(softmax(distribution));
        }
        softmaxed
    }

    pub fn get_expected_outputs(&self) -> &Vec<f64> {
        &self.expected_outputs
    }

    pub fn get_expected_distributions(&self) -> &Vec<Vec<f64>> {
        &self.expected_distribution
    }

    pub fn get_current_distributions(&self) -> &Vec<Vec<f64>> {
        &self.unsoftmaxed_distributions
    }

    pub fn get_number_dacs(&self) -> usize {
        self.dacs.len()
    }

    pub fn get_dac_i(&self, i: usize) -> &Dac {
        &self.dacs[i]
    }

    pub fn start_logger(&mut self) {
        self.log.start();
    }

    pub fn log_epoch(&mut self, loss:&Vec<f64>, lr:f64) {
        self.log.log_epoch(loss, lr, self.epsilon, self.ratio_learn);
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
    pub fn evaluate(&mut self, eval_approx:bool) -> Vec<f64> {
        let softmaxed = self.get_softmaxed_array();
        for dac in self.dacs.iter_mut() {
            dac.reset_distributions(&softmaxed);
        }
        self.dacs.par_iter_mut().for_each(|d| {
            d.reset(eval_approx);
            d.evaluate();
        });
        self.dacs.iter().map(|d| d.get_circuit_probability().to_f64()).collect::<Vec<f64>>()
    }

    // --- Gradient computation --- //

    // Compute the gradient of the distributions, from the different DAC queries
    pub fn compute_gradients(&mut self, gradient_loss: &Vec<f64>) {
        self.zero_grads();
        for dac_id in 0..self.dacs.len() {
            // Iterate on all nodes from the DAC, top-down way
            for node in self.dacs[dac_id].iter_rev() {
                let start = self.dacs[dac_id][node].get_input_start();
                let end = start + self.dacs[dac_id][node].get_number_inputs();
                let value = self.dacs[dac_id][node].get_value();
                let path_val = self.dacs[dac_id][node].get_path_value();

                // Update the path value for the children sum, product nodes 
                // and compute the gradient for the children leaf distributions
                for child_index in start..end {
                    let child = self.dacs[dac_id].get_input_at(child_index);
                    match self.dacs[dac_id][child].get_type() {
                        TypeNode::Sum => {
                            let val = path_val.clone() * &value / self.dacs[dac_id][child].get_value();
                            self.dacs[dac_id][child].set_path_value(val)
                        },
                        TypeNode::Product => {
                            self.dacs[dac_id][child].set_path_value(path_val.clone());
                        },
                        TypeNode::Distribution { d, v } => {
                            // Compute the gradient for children that are leaf distributions
                            let mut factor = path_val.clone() * gradient_loss[dac_id];
                            if let TypeNode::Product = self.dacs[dac_id][node].get_type() {
                                factor *= &value;
                                factor /= self.get_probability(d, v);
                            }
                            // Compute the gradient contribution for the value used in the node and all the other possible values of the distribution
                            let mut sum_other_w = f128!(0.0);
                            let child_w = self.get_probability(d, v);
                            for params in (0..self.unsoftmaxed_distributions[d].len()).filter(|p| *p != v) {
                                let weight = self.get_probability(d, params);
                                self.gradients[d][params] -= factor.clone() * weight.clone() * child_w.clone();
                                sum_other_w += weight.clone();
                            }
                            self.gradients[d][v] += factor * child_w * sum_other_w;
                        },
                    }
                }
            }
        }
    }

    pub fn update_distributions(&mut self) {
        for (distribution, grad) in self.unsoftmaxed_distributions.iter_mut().zip(self.gradients.iter()) {
            for (value, grad) in distribution.iter_mut().zip(grad.iter()) {
                *value -= (self.lr * grad.clone()).to_f64();
            }
        }
    }

    // --- Training --- //
    fn training_to_file(& self) {
        let mut out_writer = match &self.outfolder {
            Some(x) => {
                Box::new(File::create(x.join(format!("learn_e{}_r{}.out", self.epsilon, self.ratio_learn))).unwrap()) as Box<dyn Write>
            }
            None => Box::new(io::stdout()) as Box<dyn Write>,
        };
        writeln!(out_writer, "Obtained distributions:").unwrap();
        for i in 0..self.unsoftmaxed_distributions.len() {
            writeln!(out_writer, "Distribution {}: {:?}", i, self.get_softmaxed(i)).unwrap();
        }
        
    }

    pub fn train(&mut self, nepochs:usize, init_lr:f64, loss: Loss, timeout:i64,) {
        self.lr = init_lr;
        let lr_drop: f64 = 0.75;
        let epoch_drop = 100.0;
        let stopping_criterion = 0.0001;
        let mut prev_loss = 1.0;
        let delta_early_stop = 0.0001;
        let eval_approx_freq = 500;
        self.log.start();
        let start = chrono::Local::now();

        let mut dac_loss = vec![0.0; self.dacs.len()];
        let mut dac_grad = vec![0.0; self.dacs.len()];
        for e in 0..nepochs {
            if (chrono::Local::now() - start).num_seconds() > timeout { break;}
            let do_print = e % 500 == 0;
            self.lr = init_lr * lr_drop.powf(((1+e) as f64/ epoch_drop).floor());
            if do_print{println!("Epoch {} lr {}", e, self.lr);}
            let predictions = self.evaluate(e % eval_approx_freq == 0);
            // TODO: Maybe we can pass that in the logger and do something like self.log.print()
            if do_print { println!("--- Epoch {} ---\n Predictions: {:?} \nExpected: {:?}\n", e, predictions, self.expected_outputs);}
            /* if do_print { 
                for i in 0..self.expected_distribution.len(){
                    println!("Distribution {}  predicted {:?} expected {:?}", i, self.get_softmaxed(i), self.expected_distribution[i]);
                }
            } */
            (dac_loss, dac_grad) = loss_and_grad(loss, &predictions, &self.expected_outputs, dac_loss, dac_grad);
            self.compute_gradients(&dac_grad);
            //if do_print{ println!("Gradients: {:?}", self.gradients);}
            self.update_distributions();
            self.log.log_epoch(&dac_loss, self.lr, self.epsilon, self.ratio_learn);
            let mut avg_loss = dac_loss.iter().sum::<f64>() / dac_loss.len() as f64;
            if (avg_loss < stopping_criterion) || (avg_loss/prev_loss<delta_early_stop) {
                if self.ratio_learn < 1.0 {
                    let predictions = self.evaluate(true);
                    (dac_loss, dac_grad) = loss_and_grad(loss, &predictions, &self.expected_outputs, dac_loss, dac_grad);
                    avg_loss = dac_loss.iter().sum::<f64>() / dac_loss.len() as f64;
                    if (avg_loss < stopping_criterion) || (avg_loss/prev_loss<delta_early_stop) {
                        println!("breaking at epoch {} with avg_loss {} and prev_loss {}", e, avg_loss, prev_loss);
                        break;
                    }
                } else {
                    println!("breaking at epoch {} with avg_loss {} and prev_loss {}", e, avg_loss, prev_loss);
                    break;
                }
            }
            prev_loss = avg_loss;
            dac_loss.fill(0.0);
            dac_grad.fill(0.0);
        }

        self.training_to_file();
    }
}

// --- Indexing the graph with dac indexes --- //
impl <const S: bool> std::ops::Index<DacIndex> for Learner<S> {
    type Output = Dac;

    fn index(&self, index: DacIndex) -> &Self::Output {
        &self.dacs[index.0]
    }
}

// --- Display/Output methods ---- 

// TODO: Implementing Display for outputting the distributions is maybe not adequate, but need to
// think about that
impl <const S: bool> fmt::Display for Learner<S>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for distribution in self.expected_distribution.iter() {
            writeln!(f, "d {} {}", distribution.len(), distribution.iter().map(|p| format!("{:.5}", p)).collect::<Vec<String>>().join(" "))?;
        }
        fmt::Result::Ok(())
    }
}

impl <const S: bool> Learner<S>
{
    pub fn to_folder(&self) {
        if let Some(f) = &self.outfolder {
            let mut outfile = File::create(f.join("distributions.fdist")).unwrap();
            match outfile.write(format!("{}", self).as_bytes()) {
                Ok(_) => (),
                Err(e) => println!("Could not write the distributions into the fdist file: {:?}", e),
            }
            for (i, dac) in self.dacs.iter().enumerate() {
                let mut outfile = File::create(f.join(format!("{}.fdac", i))).unwrap();
                if let Err(e) = outfile.write(format!("{}", dac).as_bytes()) {
                    panic!("Could not write dac {} into file: {}", i, e);
                }
            }
        }
    }
}
