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
use std::io::Write;
use crate::diagrams::dac::dac::*;
use super::logger::Logger;
use search_trail::StateManager;
use crate::branching::*;
use crate::parser::*;
use crate::propagator::Propagator;
use crate::core::components::ComponentExtractor;
use crate::Branching;
use crate::{Optimizer as OptChoice, Loss};
use crate::solvers::DACCompiler;
use crate::solvers::*;
use crate::core::graph::DistributionIndex;
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
    distribution_tensors: Vec<Tensor>,
    expected_distribution: Vec<Vec<f64>>,
    expected_outputs: Vec<Tensor>,
    log: Logger<S>,
    outfolder: Option<PathBuf>,
    epsilon: f64,
    optimizer: Optimizer,
    lr: f64,
}

impl <const S: bool> TensorLearner<S>
{
    /// Creates a new learner for the inputs given. Each inputs represent a query that needs to be
    /// solved, and the expected_outputs contains, for each query, its expected probability.
    pub fn new(inputs: Vec<PathBuf>, mut expected_outputs:Vec<f64>, epsilon:f64, branching: Branching, outfolder: Option<PathBuf>, jobs:usize, optimizer: OptChoice) -> Self {
        rayon::ThreadPoolBuilder::new().num_threads(jobs).build_global().unwrap();

        let distributions = distributions_from_cnf(&inputs[0]);
        let mut dacs = inputs.par_iter().map(|input| {
            // We compile the input. This can either be a .cnf file or a fdac file.
            // If the file is a fdac file, then we read directly from it
            let mut d = match type_of_input(input) {
                FileType::CNF => {
                    println!("Compiling {}", input.to_str().unwrap());
                    // The input is a CNF file, we need to compile it from scratch
                    // First, we need to know how much distributions are needed to compute the
                    // query.
                    let mut compiler = make_compiler!(input, branching, true);
                    if epsilon > 0.0 {
                        compiler.set_partial_mode_on();
                    }
                    let dac = compile!(compiler);
                    (dac, Some(compiler))
                },
                FileType::FDAC => {
                    println!("Reading {}", input.to_str().unwrap());
                    // The query has already been compiled, we just read from the file.
                    (Some(Dac::from_file(input)), None)
                }
            };
            // We handle the compiled circuit, if present.
            if let Some(ref mut dac) = d.0.as_mut() {
                if dac.has_cutoff_nodes() {
                    // The circuit has some nodes that have been cut-off. This means that, when
                    // evaluating the circuit, they need to be solved. Hence we stock a solver
                    // for this query.
                    let solver = make_solver!(input, branching, epsilon, None, false, false);
                    dac.set_solver(solver);
                }
            }
            d
        }).collect::<Vec<_>>();
        let logger = Logger::new(outfolder.as_ref(), dacs.iter().filter(|d| d.0.is_some()).count());

        let mut is_distribution_learned = vec![false; distributions.len()];

        for (dac_opt, _) in dacs.iter() {
            if let Some(dac) = dac_opt.as_ref() {
                for (d, _) in dac.distribution_mapping.keys() {
                    is_distribution_learned[d.0] = true;
                }
            }
        }

        let mut dac_id = 0;
        for distribution in (0..is_distribution_learned.len()).map(DistributionIndex) {
            if is_distribution_learned[distribution.0] {
                continue;
            }
            // We must learn the distribution, we try to insert it into a dac
            let end = dac_id;
            loop {
                let (ref mut dac_opt, ref mut compiler_opt) = &mut dacs[dac_id];
                if let Some(dac) = dac_opt {
                    if let Some(partial_node) = dac.has_partial_node_with_distribution(distribution) {
                        let compiler = compiler_opt.as_mut().unwrap();
                        compiler.extend_partial_node_with(partial_node, dac, distribution);
                        dac_id = (dac_id + 1) % dacs.len();
                        is_distribution_learned[distribution.0] = true;
                        break;
                    }
                }
                dac_id = (dac_id + 1) % dacs.len();
                if dac_id == end {
                    break;
                }
            }
        }

        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let distribution_tensors = distributions.iter().enumerate().map(|(i, distribution)| {
            let t = Tensor::from_slice(&distribution.iter().map(|d| d.log(E)).collect::<Vec<f64>>());
            root.var_copy(&format!("Distribution {}", i+1), &t)
        }).collect::<Vec<Tensor>>();

        let optimizer = match optimizer {
            OptChoice::Adam => Adam::default().build(&vs, 1e-4).unwrap(),
            OptChoice::SGD => Sgd::default().build(&vs, 1e-4).unwrap(),
        };

        let mut s_dacs: Vec<Dac<Tensor>> = vec![];
        let mut expected: Vec<Tensor> = vec![];
        while dacs.len() > 0 {
            let (dac, _) = dacs.pop().unwrap();
            let proba = expected_outputs.pop().unwrap();
            if let Some(mut d) = dac {
                d.optimize_structure();
                s_dacs.push(d);
                expected.push(Tensor::from_f64(proba));
            }
        }

        let learner = Self { 
            dacs: s_dacs,
            distribution_tensors,
            expected_distribution: distributions,
            expected_outputs: expected,
            log: logger,
            outfolder,
            epsilon,
            optimizer,
            lr: 0.0,
        };

        learner.to_folder();
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

    pub fn get_expected_distributions(&self) -> &Vec<Vec<f64>> {
        &self.expected_distribution
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

    pub fn log_epoch(&mut self, loss:&Vec<f64>, lr:f64) {
        self.log.log_epoch(loss, lr, self.epsilon);
    }
    // --- Evaluation --- //

    // Evaluate the different DACs and return the results
    fn evaluate(&mut self, eval_approx:bool) {
        for i in 0..self.dacs.len() {
            let softmaxed = self.get_softmaxed_array();
            self.dacs[i].reset_distributions(&softmaxed);
        }
        self.dacs.par_iter_mut().for_each(|d| {
            if eval_approx {
                d.reset_approx(eval_approx);
            }
            d.evaluate();
        });
    }
}

impl<const S: bool> Learning for TensorLearner<S> {

    fn train(&mut self, nepochs:usize, init_lr:f64, loss: Loss, timeout:i64,) {
        self.lr = init_lr;
        self.optimizer.set_lr(self.lr);
        let lr_drop: f64 = 0.75;
        let epoch_drop = 100.0;
        let stopping_criterion = 0.0001;
        let mut prev_loss = 1.0;
        let delta_early_stop = 0.00001;
        let eval_approx_freq = 500;
        let mut count_no_improve = 0;
        let patience = 5;
        self.log.start();
        let start = chrono::Local::now();

        let mut dac_loss = vec![0.0; self.dacs.len()];
        for e in 0..nepochs {
            if (chrono::Local::now() - start).num_seconds() > timeout { break;}
            let do_print = e % 1000 == 0;
            self.lr = init_lr * lr_drop.powf(((1+e) as f64/ epoch_drop).floor());
            self.optimizer.set_lr(self.lr);
            if do_print{println!("\nEpoch {} lr {}", e, self.lr);}
            self.evaluate(e % eval_approx_freq == 0);
            if do_print {
                for i in 0..self.dacs.len() {
                    println!("{} {} {}", i, self.dacs[i].root().to_f64(), self.expected_outputs[i].to_f64());
                }
            }
            let mut loss_epoch = Tensor::from(0.0);
            for i in 0..self.dacs.len() {
                let loss_i = match loss {
                    Loss::MAE => self.dacs[i].root().l1_loss(&self.expected_outputs[i], Reduction::Mean),
                    Loss::MSE => self.dacs[i].root().mse_loss(&self.expected_outputs[i], Reduction::Mean),
                };
                dac_loss[i] = loss_i.to_f64();
                loss_epoch += loss_i;
            }
            //loss_epoch /= self.dacs.len() as f64;
            self.optimizer.backward_step(&loss_epoch);
            self.log.log_epoch(&dac_loss, 0.0, self.epsilon);
            let mut avg_loss = dac_loss.iter().sum::<f64>() / dac_loss.len() as f64;
            if (avg_loss-prev_loss).abs()<delta_early_stop {
                count_no_improve += 1;
            }
            else {
                count_no_improve = 0;
            }
            if (avg_loss < stopping_criterion) || count_no_improve>=patience {
                // TODO: Before we checked if the learned ratio was < 1.0 and launched only in that
                // case the additional evaluation. Should probabily do something of the like with
                // the cut-off nodes int he DACs?
                self.evaluate(true);
                let mut loss_epoch = Tensor::from(0.0);
                for i in 0..self.dacs.len() {
                    let loss_i = match loss {
                        Loss::MAE => self.dacs[i].root().l1_loss(&self.expected_outputs[i], Reduction::Mean),
                        Loss::MSE => self.dacs[i].root().mse_loss(&self.expected_outputs[i], Reduction::Mean),
                    };
                    dac_loss[i] = loss_i.to_f64();
                    loss_epoch += loss_i;
                }
                avg_loss = dac_loss.iter().sum::<f64>() / dac_loss.len() as f64;
                if (avg_loss-prev_loss).abs()>delta_early_stop {
                    count_no_improve = 0;
                }
                if avg_loss < stopping_criterion || count_no_improve>=patience{
                    println!("breaking at epoch {} with avg_loss {} and prev_loss {}", e, avg_loss, prev_loss);
                    break;
                }
            }
            prev_loss = avg_loss;
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

// --- Display/Output methods ---- 

// TODO: Implementing Display for outputting the distributions is maybe not adequate, but need to
// think about that
impl <const S: bool> fmt::Display for TensorLearner<S>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for distribution in self.expected_distribution.iter() {
            writeln!(f, "d {} {}", distribution.len(), distribution.iter().map(|p| format!("{:.5}", p)).collect::<Vec<String>>().join(" "))?;
        }
        fmt::Result::Ok(())
    }
}

impl <const S: bool> TensorLearner<S>
{
    pub fn to_folder(&self) {
        /*if let Some(f) = &self.outfolder {
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
        }*/
    }
}
