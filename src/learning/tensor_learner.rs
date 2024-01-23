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
use crate::diagrams::dac::dac::*;
use super::logger::Logger;
use search_trail::StateManager;
use crate::branching::*;
use crate::parser::*;
use crate::propagator::Propagator;
use crate::core::components::ComponentExtractor;
use crate::Branching;
use crate::{Optimizer as OptChoice, Loss};
use crate::solvers::*;
use rayon::prelude::*;
use super::Learning;
use std::f64::consts::E;
use crate::diagrams::semiring::SemiRing;

use tch::{Kind, nn, Device, Tensor, IndexOp, Reduction};
use tch::nn::{OptimizerConfig, Adam, Sgd, Optimizer};


/// Abstraction used as a typesafe way of retrieving a `DAC` in the `Learner` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DacIndex(pub usize);

pub struct TensorLearner<const S: bool>
{
    dacs: Vec<Dac<Tensor>>,
    test_dacs: Vec<Dac<Tensor>>,
    distribution_tensors: Vec<Tensor>,
    expected_outputs: Vec<Tensor>,
    test_expected_outputs: Vec<Tensor>,
    log: Logger<S>,
    test_log: Logger<S>,
    epsilon: f64,
    optimizer: Optimizer,
    lr: f64,
    // TODO a field for distributions that are learned -> not backpropagate them
}

impl <const S: bool> TensorLearner<S>
{
    /// Creates a new learner for the inputs given. Each inputs represent a query that needs to be
    /// solved, and the expected_outputs contains, for each query, its expected probability.
    pub fn new(inputs: Vec<PathBuf>, mut expected_outputs:Vec<f64>, epsilon:f64, branching: Branching, outfolder: Option<PathBuf>, jobs:usize, optimizer: OptChoice, test_inputs:Vec<PathBuf>, mut test_expected:Vec<f64>) -> Self {
        rayon::ThreadPoolBuilder::new().num_threads(jobs).build_global().unwrap();

        let distributions = distributions_from_cnf(&inputs[0]);
        let mut dacs = inputs.par_iter().map(|input| {
            // We compile the input. This can either be a .cnf file or a fdac file.
            // If the file is a fdac file, then we read directly from it
            match type_of_input(input) {
                FileType::CNF => {
                    println!("Compiling {}", input.to_str().unwrap());
                    // The input is a CNF file, we need to compile it from scratch
                    // First, we need to know how much distributions are needed to compute the
                    // query.
                    let compiler = make_solver!(input, branching, epsilon, None, false);
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
        let mut test_dacs = test_inputs.par_iter().map(|input| {
            // We compile the input. This can either be a .cnf file or a fdac file.
            // If the file is a fdac file, then we read directly from it
            match type_of_input(input) {
                FileType::CNF => {
                    println!("Compiling {}", input.to_str().unwrap());
                    // The input is a CNF file, we need to compile it from scratch
                    // First, we need to know how much distributions are needed to compute the
                    // query.
                    let compiler = make_solver!(input, branching, epsilon, None, false);
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
        let logger = Logger::new(outfolder.as_ref(), dacs.iter().filter(|d| d.is_some()).count(), false);
        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let distribution_tensors = distributions.iter().enumerate().map(|(i, distribution)| {
            let t = Tensor::from_slice(&distribution.iter().map(|d| d.log(E)).collect::<Vec<f64>>()).requires_grad_(true);
            root.var_copy(&format!("Distribution {}", i+1), &t)
        }).collect::<Vec<Tensor>>();

        let optimizer = match optimizer {
            OptChoice::Adam => Adam::default().build(&vs, 1e-4).unwrap(),
            OptChoice::SGD => Sgd::default().build(&vs, 1e-4).unwrap(),
        };

        let mut s_dacs: Vec<Dac<Tensor>> = vec![];
        let mut expected: Vec<Tensor> = vec![];
        while dacs.len() > 0 {
            let dac = dacs.pop().unwrap();
            let proba = expected_outputs.pop().unwrap();
            if let Some(d) = dac {
                //println!("dac\n{}", d.as_graphviz());
                s_dacs.push(d);
                expected.push(Tensor::from_f64(proba));
            }
        }
        let mut test_s_dacs: Vec<Dac<Tensor>> = vec![];
        let mut test_expected_o: Vec<Tensor> = vec![];
        while test_dacs.len() > 0 {
            let dac = test_dacs.pop().unwrap();
            let proba = test_expected.pop().unwrap();
            if let Some(d) = dac {
                //println!("dac\n{}", d.as_graphviz());
                test_s_dacs.push(d);
                test_expected_o.push(Tensor::from_f64(proba));
            }
        }

        let mut learner = Self { 
            dacs: s_dacs,
            test_dacs: test_s_dacs,
            distribution_tensors,
            expected_outputs: expected,
            test_expected_outputs: test_expected_o,
            log: logger,
            test_log: Logger::default(),
            epsilon,
            optimizer,
            lr: 0.0,
        };
        if test_inputs.len()!=0{ learner.test_log = Logger::new(outfolder.as_ref(), learner.dacs.len(), false);}
        learner
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

    pub fn get_number_dacs(&self) -> usize {
        self.dacs.len()
    }

    pub fn get_dac_i(&self, i: usize) -> &Dac<Tensor> {
        &self.dacs[i]
    }

    pub fn start_logger(&mut self) {
        self.log.start();
    }

    pub fn log_epoch(&mut self, loss:&Vec<f64>, lr:f64, predictions:&Vec<f64>) {
        self.log.log_epoch(loss, lr, self.epsilon, predictions);
    }
    // --- Evaluation --- //

    // Evaluate the different DACs and return the results
    fn evaluate(&mut self) -> Vec<f64> {
        for i in 0..self.dacs.len() {
            let softmaxed = self.get_softmaxed_array();
            self.dacs[i].reset_distributions(&softmaxed);
        }
        self.dacs.par_iter_mut().for_each(|d| {
            d.evaluate();
        });
        self.dacs.iter().map(|d| d.circuit_probability().to_f64()).collect()
    }

    // Evaluate the different test DACs and return the results
    fn test(&mut self) -> Vec<f64> {
        for i in 0..self.test_dacs.len() {
            let softmaxed = self.get_softmaxed_array();
            self.dacs[i].reset_distributions(&softmaxed);
        }
        self.test_dacs.par_iter_mut().for_each(|d| {
            d.evaluate();
        });
        self.test_dacs.iter().map(|d| d.circuit_probability().to_f64()).collect()
    }
}

impl<const S: bool> Learning for TensorLearner<S> {

    fn train(&mut self, nepochs:usize, init_lr:f64, loss: Loss, timeout:u64,) {
        self.lr = init_lr;
        self.optimizer.set_lr(self.lr);
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
        for e in 0..nepochs {
            if (chrono::Local::now() - start).num_seconds() > timeout as i64 { break;}
            let do_print = e % 500 == 0;
            self.lr = init_lr * lr_drop.powf(((1+e) as f64/ epoch_drop).floor());
            self.optimizer.set_lr(self.lr);
            if do_print{println!("\nEpoch {} lr {}", e, self.lr);}
            let predictions = self.evaluate();
            if do_print {
                for i in 0..self.dacs.len() {
                    println!("{} {} {}", i, self.dacs[i].circuit_probability().to_f64(), self.expected_outputs[i].to_f64());
                }
            }
            let mut loss_epoch = Tensor::from(0.0);
            for i in 0..self.dacs.len() {
                let loss_i = match loss {
                    Loss::MAE => self.dacs[i].circuit_probability().l1_loss(&self.expected_outputs[i], Reduction::Mean),
                    Loss::MSE => self.dacs[i].circuit_probability().mse_loss(&self.expected_outputs[i], Reduction::Mean),
                };
                dac_loss[i] = loss_i.to_f64();
                loss_epoch += loss_i;
            }
            //loss_epoch /= self.dacs.len() as f64;
            self.optimizer.backward_step(&loss_epoch);
            /* for d in self.distribution_tensors.iter() {
                println!("tensor grad: {:?}", d.grad());
            } */
            self.log.log_epoch(&dac_loss, 0.0, self.epsilon, &predictions);
            let avg_loss = dac_loss.iter().sum::<f64>() / dac_loss.len() as f64;
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
        }
        if self.test_dacs.len()!=0{
            self.test_log.start();
            let predictions = self.test();
            let mut loss_epoch = Tensor::from(0.0);
            let mut dac_loss = vec![0.0; self.test_dacs.len()];
            for i in 0..self.test_dacs.len() {
                let loss_i = match loss {
                    Loss::MAE => self.test_dacs[i].circuit_probability().l1_loss(&self.test_expected_outputs[i], Reduction::Mean),
                    Loss::MSE => self.test_dacs[i].circuit_probability().mse_loss(&self.test_expected_outputs[i], Reduction::Mean),
                };
                dac_loss[i] = loss_i.to_f64();
                loss_epoch += loss_i;
                self.test_log.log_epoch(&dac_loss, self.lr, self.epsilon, &predictions);
            }
        }
    }
}

// --- Indexing the graph with dac indexes --- //
impl <const S: bool> std::ops::Index<DacIndex> for TensorLearner<S> 
{
    type Output = Dac<Tensor>;

    fn index(&self, index: DacIndex) -> &Self::Output {
        &self.dacs[index.0]
    }
}

