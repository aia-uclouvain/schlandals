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
use search_trail::StateManager;
use std::time::{Instant, Duration};
use crate::ac::ac::*;
use crate::ac::node::NodeType;
use super::logger::Logger;
use crate::*;
use rayon::prelude::*;
use crate::common::F128;
use super::*;
use rug::{Assign, Float};
use std::ffi::OsString;
use crate::parse_csv;
use crate::semiring::SemiRing;

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
    epsilon: f64,
    clauses: Vec<Vec<Vec<isize>>>,
    learning_m: LearningMethod,
}

impl <const S: bool> Learner<S> {
    pub fn new(input: PathBuf, args: Args) -> Self {
        if let Command::Learn { trainfile,
                                testfile,
                                outfolder,
                                lr: _,
                                nepochs: _,
                                do_log: _,
                                ltimeout: _,
                                loss: _,
                                jobs,
                                semiring: _,
                                optimizer: _,
                                lr_drop: _,
                                epoch_drop: _,
                                early_stop_threshold: _,
                                early_stop_delta: _,
                                patience: _, 
                                equal_init,
                                recompile: _,
                                e_weighted: _,
                                learning_m,
                                lds_opti,
                            } = args.subcommand.unwrap() {
            if jobs!=1{rayon::ThreadPoolBuilder::new().num_threads(jobs).build_global().unwrap();}
        
            // Parsing the input files to retrieve the problem and the queries
            let parser = parser_from_input(input.clone(), Some(OsString::default()));
            let mut train_queries = parse_csv(trainfile);
            let mut test_queries = if let Some(file) = testfile { parse_csv(file) } else { vec![] };
            //println!("Test queries: {}", test_queries.len());
            let mut clauses = vec![];
            let mut all_distributions = vec![];
            let mut test_clauses = vec![];
            let mut all_test_distributions = vec![];
            for (query, _) in train_queries.iter() {
                let q_parser = parser_from_input(PathBuf::from(query.clone()), Some(OsString::default())); //parser_from_input(input.clone(), Some(query.clone()));
                let c = q_parser.clauses_from_file();
                let d = q_parser.distributions_from_file();
                clauses.push(c);
                all_distributions.push(d);
            }
            for (query, _) in test_queries.iter() {
                let q_parser = parser_from_input(PathBuf::from(query.clone()), Some(OsString::default())); //parser_from_input(input.clone(), Some(query.clone()));
                let c = q_parser.clauses_from_file();
                let d = q_parser.distributions_from_file();
                test_clauses.push(c);
                all_test_distributions.push(d);
            }

            // Retrieves the distributions values and computes their unsoftmaxed values
            // and initializes the gradients to 0
            let distributions = parser.distributions_from_file();
            let mut grads: Vec<Vec<Float>> = vec![];
            let mut unsoftmaxed_distributions: Vec<Vec<f64>> = vec![];
            let eps = args.epsilon;
            for distribution in distributions.iter() {
                if !equal_init{
                    let unsoftmaxed_vector = distribution.iter().map(|p| p.log(std::f64::consts::E)).collect::<Vec<f64>>();
                    unsoftmaxed_distributions.push(unsoftmaxed_vector);
                }
                else {
                    let unsoftmaxed_vector = distribution.iter().map(|_| 1.0/(distribution.len() as f64).log(std::f64::consts::E)).collect::<Vec<f64>>();
                    unsoftmaxed_distributions.push(unsoftmaxed_vector);
                }
                grads.push(vec![F128!(0.0); distribution.len()]);
            }

            // Compiling the train and test queries into arithmetic circuits
            //println!("Train");
            let c_time = Instant::now();
            let mut train_dacs: Vec<Dac<Float>> = generate_dacs(&clauses, &all_distributions, args.branching, args.epsilon, args.approx, args.timeout, learning_m, lds_opti);
            let c_duration = c_time.elapsed().as_secs();
            //println!("Test");
            let mut test_dacs: Vec<Dac<Float>> = generate_dacs(&test_clauses, &all_test_distributions, args.branching, args.epsilon, args.approx, args.timeout, LearningMethod::Models, false);
            let mut train_dataset = Dataset::<Float>::new(vec![], vec![]);
            let mut test_dataset = Dataset::<Float>::new(vec![], vec![]);
            let mut cnt = 0;
            if learning_m == LearningMethod::Both {
                let mut q1 = train_queries.clone();
                let mut q2 = train_queries.clone();
                train_queries = vec![];
                while !q1.is_empty() {
                    train_queries.push(q2.pop().unwrap());
                    train_queries.push(q1.pop().unwrap());                    
                }
                train_queries.reverse();
            }
            while let Some(mut d) = train_dacs.pop() {
                let expected = match learning_m {
                    LearningMethod::Models => train_queries.pop().unwrap().1,
                    LearningMethod::NonModels => 1.0 - train_queries.pop().unwrap().1,
                    LearningMethod::Both => if cnt % 2 != 0 {train_queries.pop().unwrap().1} else {1.0 - train_queries.pop().unwrap().1},
                };
                if args.epsilon == 0.0 && d.epsilon() > 0.00001 && args.approx != ApproximateMethod::LDS {
                    println!("Ignoring query with epsilon {} as compilation timeout", d.epsilon());
                    continue;
                }
                d.evaluate();
                train_dataset.add_query(d,expected);
                cnt += 1;
            }
            train_dataset.reverse(); // We reverse the queries to have the same order as the input file (and first the model followed by the non-model in case of learning both)
            while let Some(mut d) = test_dacs.pop() {
                let expected = test_queries.pop().unwrap().1;
                d.evaluate();
                //println!("eval {} expected {}", d.circuit_probability().to_f64(), expected);
                test_dataset.add_query(d, expected);
            }
            test_dataset.reverse();
            // Initializing the logger
            let log = Logger::new(outfolder.as_ref(), train_dataset.len(), test_dataset.len(), learning_m, c_duration as usize);
            Self {
                train: train_dataset,
                test: test_dataset,
                unsoftmaxed_distributions, 
                gradients: grads,
                log,
                epsilon: eps,
                clauses,
                learning_m,
            }
        } else {
            panic!("learning procedure called with non-learn command line arguments");
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

     /// Return the gradients of the parameters
     pub fn get_gradients(&self) -> &Vec<Vec<Float>> {
        &self.gradients
    }

    /// Return the train dataset
    pub fn get_train(&self) -> &Dataset<Float> {
        &self.train
    }

    /// Return the test dataset
    pub fn get_test(&self) -> &Dataset<Float> {
        &self.test
    }

    /* /// Return the training queries, or an empty vector if no queries were provided
    pub fn get_queries(&self) -> &Vec<(f64,f64)> {
        &self.queries
    } */

    pub fn array_f64_to_f128(&self, array: Vec<Vec<f64>>) -> Vec<Vec<Float>> {
        array.iter().map(|row| row.iter().map(|&x| F128!(x)).collect()).collect()
    }

    // --- Setters --- //

    /// Set the distributions to the given values (that are softmaxed or unsoftmaxed)
    pub fn set_distributions(&mut self, distributions: Vec<Vec<f64>>, is_softmaxed: bool) {
        if is_softmaxed{
            self.unsoftmaxed_distributions = distributions.iter().map(|row| row.iter().map(|&x| x.ln()).collect()).collect();
        } else {
            self.unsoftmaxed_distributions = distributions;
        }
    }

    pub fn set_gradients(&mut self, gradients: Vec<Vec<Float>>) {
        self.gradients = gradients;
    }

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

    fn recompile_dacs(&mut self, branching: Branching, approx:ApproximateMethod, compile_timeout: u64, learning_m: LearningMethod, lds_opti:bool) {
        let distributions: Vec<Vec<f64>> = self.get_softmaxed_array().iter().map(|d| d.iter().map(|f| f.to_f64()).collect::<Vec<f64>>()).collect();
        let repeat_distributions = vec![distributions.clone(); self.train.len()];
        let mut train_dacs = generate_dacs(&self.clauses, &repeat_distributions, branching, self.epsilon, approx, compile_timeout, learning_m, lds_opti);
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
    pub fn compute_gradients(&mut self, gradient_loss: &[f64], idxs: Option<Vec<usize>>) -> Vec<Vec<f64>> {
        self.zero_grads();
        let ids = idxs.unwrap_or((0..self.train.len()).collect());
        let z_g = self.get_gradients().clone(); 
        let mut current_g = self.get_gradients().clone();
        let mut grads = vec![];
        for query_id in ids {
            if self.learning_m == LearningMethod::Both { current_g = z_g.clone();}
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
                                val = path_val.clone() * value / self.train[query_id][child].value().to_f64();
                            }
                            self.train[query_id][child].add_to_path_value(val);
                        },
                        NodeType::Sum => {
                            // If it is a sum node, we simply propagate the path value to the children
                            self.train[query_id][child].add_to_path_value(path_val.clone());
                        },
                        NodeType::Distribution { .. } => {},
                        NodeType::Opposite => {
                            // If it is an opposite node, we need to negate the path value
                            self.train[query_id][child].add_to_path_value(-path_val.clone());
                        },
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
                            if self.learning_m == LearningMethod::Both {
                                current_g[d][params] -= factor.clone() * weight * child_w;
                            }
                            sum_other_w += weight;
                        }
                        if self.learning_m == LearningMethod::Both {
                            current_g[d][v] += factor.clone() * child_w * sum_other_w.clone();
                        }
                        self.gradients[d][v] += factor * child_w * sum_other_w;
                    }
                }
            }
            if self.learning_m == LearningMethod::Both {
                let mut cnt = vec![];
                for i in 0..current_g.len(){
                    for j in 0..current_g[i].len(){
                            cnt.push(current_g[i][j].clone().to_f64());
                    }
                }
                grads.push(cnt);
            }
            else {
                grads.push(vec![]);
            }
        }
        grads
    }

    /// Update the distributions with the computed gradients and the learning rate, following an SGD approach
    pub fn update_distributions(&mut self, learning_rate: f64) {
        for (distribution, grad) in self.unsoftmaxed_distributions.iter_mut().zip(self.gradients.iter()) {
            for (value, grad) in distribution.iter_mut().zip(grad.iter()) {
                *value -= (learning_rate * grad.clone()).to_f64();
            }
        }
    }

    /// Training loop for the train dacs, using the given training parameters
    //fn train(&mut self, params: &LearnParameters, inputs: &Vec<PathBuf>, branching: Branching, approx:ApproximateMethod, compile_timeout: u64) {
    pub fn train(&mut self, params: &LearnParameters, branching: Branching, approx:ApproximateMethod) {
        let mut prev_loss = 1.0;
        let mut count_no_improve = 0;
        self.log.start();
        let start = Instant::now();
        let to = Duration::from_secs(params.learning_timeout());
        
        // Evaluate the test set before training, if it exists
        if self.test.len() != 0 {
            let predictions = self.test();
            let test_loss = predictions.iter().copied().enumerate().map(|(i, prediction)| params.loss().loss(prediction, self.test.expected(i))).collect::<Vec<f64>>();
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
            let mut other_pred = vec![];
            // For experiments on the loss computation
            if false && self.learning_m == LearningMethod::Both {
                for i in 0..predictions.len()/2 {
                    other_pred.push((predictions[2*i]*(1.0-predictions[2*i+1])).sqrt());
                    other_pred.push(1.0-(predictions[2*i]*(1.0-predictions[2*i+1])).sqrt());
                }
            }
            else {
                other_pred = predictions.clone();
            }
            // Compute the loss and the gradients
            for i in 0..predictions.len() {
                train_loss[i] = params.loss().loss(predictions[i], self.train.expected(i));
                train_grad[i] = params.loss().gradient(other_pred[i], self.train.expected(i));//predictions[i], self.train.expected(i));
                if params.e_weighted() && self.epsilon != 0.0 {
                    let l = self.train.len() as f64;
                    train_grad[i] *= (1.0 - self.train[i].epsilon()/self.epsilon) * l / (l - 1.0);
                }
            }
            let avg_loss = train_loss.iter().sum::<f64>() / train_loss.len() as f64;
            let grads = self.compute_gradients(&train_grad, None);
            self.log.log_epoch(&train_loss, learning_rate, self.epsilon, &predictions, &grads, &self.get_softmaxed_array());
            // Update the parameters
            self.update_distributions(learning_rate);
            
            // TODO Remove
            if self.test.len() != 0 {
                let predictions = self.test();
                let test_loss = predictions.iter().copied().enumerate().map(|(i, prediction)| params.loss().loss(prediction, self.test.expected(i))).collect::<Vec<f64>>();
                self.log.log_test(&test_loss, self.epsilon, &predictions);
            }
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

            if e != params.nepochs() - 1 && params.recompile() {
                println!("Recompiling at epoch {}", e);
                self.recompile_dacs(branching, approx, params.compilation_timeout(), self.learning_m, params.lds_opti());
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
            let test_loss = predictions.iter().copied().enumerate().map(|(i, prediction)| params.loss().loss(prediction, self.test.expected(i))).collect::<Vec<f64>>();
            self.log.log_test(&test_loss, self.epsilon, &predictions);
        }
    }
}

impl <const S: bool> std::ops::Index<DacIndex> for Learner<S> {
    type Output = Dac<Float>;

    fn index(&self, index: DacIndex) -> &Self::Output {
        &self.train[index.0]
    }
}

/// Calculates the softmax (the normalized exponential) function, which is a generalization of the
/// logistic function to multiple dimensions.
///
/// Takes in a vector of real numbers and normalizes it to a probability distribution such that
/// each of the components are in the interval (0, 1) and the components add up to 1. Larger input
/// components correspond to larger probabilities.
/// From https://docs.rs/compute/latest/src/compute/functions/statistical.rs.html#43-46
pub fn softmax(x: &[f64]) -> Vec<Float> {
    let sum_exp: f64 = x.iter().map(|i| i.exp()).sum();
    x.iter().map(|i| F128!(i.exp() / sum_exp)).collect()
}

/// Generates a vector of optional Dacs from a list of input files
pub fn generate_dacs<R: SemiRing>(queries_clauses: &Vec<Vec<Vec<isize>>>, distributions: &Vec<Vec<Vec<f64>>>,branching: Branching, epsilon: f64, 
    approx: ApproximateMethod, timeout: u64, learning_m: LearningMethod, lds_opti:bool) -> Vec<Dac<R>> {
    let mut double_dacs = queries_clauses.par_iter().enumerate().map(|(i, clauses)| {
        // We compile the input. This can either be a .cnf file or a fdac file.
        // If the file is a fdac file, then we read directly from it
        let mut state = StateManager::default();
        let problem = create_problem(&distributions[i], clauses, &mut state); //parser.problem_from_file(&mut state);
        let parameters = SolverParameters::new(u64::MAX, epsilon, timeout);
        let propagator = Propagator::new(&mut state);
        let component_extractor = ComponentExtractor::new(&problem, &mut state);
        let compiler = generic_solver(problem, state, component_extractor, branching, propagator, parameters, false);
        match approx {
            ApproximateMethod::Bounds => {
                match compiler {
                    crate::GenericSolver::SMinInDegree(mut s) => s.compile(false, lds_opti),
                    crate::GenericSolver::QMinInDegree(mut s) => s.compile(false, lds_opti),
                    crate::GenericSolver::SMinConstrained(mut s) => s.compile(false, lds_opti),
                    crate::GenericSolver::QMinConstrained(mut s) => s.compile(false, lds_opti),
                    crate::GenericSolver::SMaxConstrained(mut s) => s.compile(false, lds_opti),
                    crate::GenericSolver::QMaxConstrained(mut s) => s.compile(false, lds_opti),
                }
            },
            ApproximateMethod::LDS => {
                match compiler {
                    crate::GenericSolver::SMinInDegree(mut s) => s.compile(true, lds_opti),
                    crate::GenericSolver::QMinInDegree(mut s) => s.compile(true, lds_opti),
                    crate::GenericSolver::SMinConstrained(mut s) => s.compile(true, lds_opti),
                    crate::GenericSolver::QMinConstrained(mut s) => s.compile(true, lds_opti),
                    crate::GenericSolver::SMaxConstrained(mut s) => s.compile(true, lds_opti),
                    crate::GenericSolver::QMaxConstrained(mut s) => s.compile(true, lds_opti),
                }
            },
        }
    }).collect::<Vec<_>>();
    let mut dacs = vec![];
    while !double_dacs.is_empty() {
        let (d1, d2) = double_dacs.pop().unwrap();
        //println!("DACs generated: {} {}", d1.number_nodes(), d2.number_nodes());
        match learning_m {
            LearningMethod::Models => dacs.push(d1),
            LearningMethod::NonModels => dacs.push(d2),
            LearningMethod::Both => {
                dacs.push(d2);
                dacs.push(d1);
            }             
        }
    }
    dacs.reverse();
    dacs
}

/// Decides whether early stopping should be performed or not
pub fn do_early_stopping(avg_loss:f64, prev_loss:f64, count:&mut usize, stopping_criterion:f64, patience:usize, delta:f64) -> bool {
    if (avg_loss-prev_loss).abs()<delta {
        *count += 1;
    }
    else {
        *count = 0;
    }
    avg_loss < stopping_criterion || *count >= patience
}

/// Structure representing a dataset for the learners. A dataset is a set of queries (boolean
/// formulas compiled into an arithmetic circuit) associated with an expected probability
#[derive(Default)]
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

    /// Adds a query to the dataset
    pub fn add_query(&mut self, query: Dac<R>, expected: f64) {
        self.queries.push(query);
        self.expected.push(expected);
    }

    /// Returns the expected output for the required query
    pub fn expected(&self, query_index: usize) -> f64 {
        self.expected[query_index]
    }

    /// Returns the list of expected outputs
    pub fn get_expecteds(&self) -> &Vec<f64> {
        &self.expected
    }

    /// Sets the queries of the dataset
    pub fn set_queries(&mut self, queries: Vec<Dac<R>>) {
        self.queries = queries;
    }

    /// Reverses the queries of the dataset
    pub fn reverse(&mut self) {
        self.queries.reverse();
        self.expected.reverse();
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
