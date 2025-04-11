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
use crate::common::rational;
use super::*;
use malachite::Rational;
use malachite::num::arithmetic::traits::Pow;
use std::ffi::OsString;
use crate::parse_csv;

/// Abstraction used as a typesafe way of retrieving a `DAC` in the `Learner` structure
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DacIndex(pub usize);

/// Structure used to learn the distribution parameters from a set of queries
pub struct Learner<const S: bool> {
    train: Dataset,
    test: Dataset,
    unsoftmaxed_distributions: Vec<Vec<Rational>>,
    gradients: Vec<Vec<Rational>>,
    log: Logger<S>,
    epsilon: f64,
    clauses: Vec<Vec<Vec<isize>>>,
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
                                optimizer: _,
                                lr_drop: _,
                                epoch_drop: _,
                                early_stop_threshold: _,
                                early_stop_delta: _,
                                patience: _, 
                                equal_init,
                                recompile: _,
                                e_weighted: _,
                                } = args.subcommand.unwrap() {
            rayon::ThreadPoolBuilder::new().num_threads(jobs).build_global().unwrap();
        
            // Parsing the input files to retrieve the problem and the queries
            let parser = parser_from_input(input.clone(), Some(OsString::default()));
            let mut train_queries = parse_csv(trainfile);
            let mut test_queries = if let Some(file) = testfile { parse_csv(file) } else { vec![] };
            let mut clauses = vec![];
            let mut test_clauses = vec![];
            for (query, _) in train_queries.iter() {
                let q_parser = parser_from_input(input.clone(), Some(query.clone()));
                /* let mut state = StateManager::default();
                let problem = q_parser.problem_from_file(&mut state);
                let c: Vec<Vec<isize>> = problem.clauses().iter().map(|c| c.iter().map(|l| 
                    if l.is_positive() {(problem[l.to_variable()].old_index() + 1) as isize}
                    else {(problem[l.to_variable()].old_index() + 1) as isize * -1}).collect()).collect(); */
                let c = q_parser.clauses_from_file();
                clauses.push(c);
            }
            for (query, _) in test_queries.iter() {
                let q_parser = parser_from_input(input.clone(), Some(query.clone()));
                /* let mut state = StateManager::default();
                let problem = q_parser.problem_from_file(&mut state);
                let c: Vec<Vec<isize>> = problem.clauses().iter().map(|c| c.iter().map(|l| 
                    if l.is_positive() {(problem[l.to_variable()].old_index() + 1) as isize}
                    else {(problem[l.to_variable()].old_index() + 1) as isize * -1}).collect()).collect(); */
                let c = q_parser.clauses_from_file();
                test_clauses.push(c);
            }

            // Retrieves the distributions values and computes their unsoftmaxed values
            // and initializes the gradients to 0
            let distributions = parser.distributions_from_file();
            let mut grads: Vec<Vec<Rational>> = vec![];
            let mut unsoftmaxed_distributions: Vec<Vec<Rational>> = vec![];
            let mut eps = args.epsilon;
            for distribution in distributions.iter() {
                if !equal_init{
                    let unsoftmaxed_vector = distribution.iter().map(|p| rational(p.approx_log())).collect::<Vec<Rational>>();
                    unsoftmaxed_distributions.push(unsoftmaxed_vector);
                }
                else {
                    let unsoftmaxed_vector = distribution.iter().map(|_| rational(1.0/(distribution.len() as f64).log(std::f64::consts::E))).collect::<Vec<Rational>>();
                    unsoftmaxed_distributions.push(unsoftmaxed_vector);
                }
                grads.push(vec![rational(0.0); distribution.len()]);
            }

            // Compiling the train and test queries into arithmetic circuits
            let mut train_dacs = generate_dacs(&clauses, &distributions, args.branching, args.epsilon, args.approx, args.timeout);
            if args.approx == ApproximateMethod::LDS {
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
            let mut test_dacs = generate_dacs(&test_clauses, &distributions, args.branching, args.epsilon, ApproximateMethod::Bounds, u64::MAX);
            let mut train_dataset = Dataset::new(vec![], vec![]);
            let mut test_dataset = Dataset::new(vec![], vec![]);
            while let Some(d) = train_dacs.pop() {
                let expected = train_queries.pop().unwrap().1;
                train_dataset.add_query(d,expected);
            }
            while let Some(d) = test_dacs.pop() {
                let expected = test_queries.pop().unwrap().1;
                test_dataset.add_query(d, expected);
            }
            // Initializing the logger
            let log = Logger::new(outfolder.as_ref(), train_dataset.len(), test_dataset.len());
            Self {
                train: train_dataset,
                test: test_dataset,
                unsoftmaxed_distributions, 
                gradients: grads,
                log,
                epsilon: eps,
                clauses,
            }
        } else {
            panic!("learning procedure called with non-learn command line arguments");
        }
    }

    // --- Getters --- //

    /// Return the softmax values for the paramater of the given distribution
    fn get_softmaxed(&self, distribution: usize) -> Vec<Rational> {
        softmax(&self.unsoftmaxed_distributions[distribution])
    }

    /// Return the probability of the given value for the given distribution
    pub fn get_probability(&self, distribution: usize, index: usize) -> Rational {
        self.get_softmaxed(distribution)[index].clone()
    }

    /// Returns a double vector of tensors. Each entry (d,i) is a tensor representing the softmaxed
    /// version of the i-th value of vector d
    pub fn get_softmaxed_array(&self) -> Vec<Vec<Rational>> {
        let mut softmaxed: Vec<Vec<Rational>> = vec![];
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
                *el = rational(0.0);
            }
        }
    }

    // --- Evaluation --- //

    // TODO: Same code, should not be duplicated
    // Evaluate the different train DACs and return the results
    pub fn evaluate(&mut self) -> Vec<Rational> {
        let softmaxed = self.get_softmaxed_array();
        for dac in self.train.get_queries_mut() {
            dac.reset_distributions(&softmaxed);
        }
        self.train.get_queries_mut().par_iter_mut().for_each(|d| {
            d.evaluate();
        });
        self.train.get_queries().iter().map(|d| d.circuit_probability()).collect()
    }

    // Evaluate the different test DACs and return the results
    pub fn test(&mut self) -> Vec<Rational> {
        let softmaxed = self.get_softmaxed_array();
        for dac in self.test.get_queries_mut() {
            dac.reset_distributions(&softmaxed);
        }
        self.test.get_queries_mut().par_iter_mut().for_each(|d| {
            d.evaluate();
        });
        self.test.get_queries().iter().map(|d| d.circuit_probability()).collect()
    }

    fn recompile_dacs(&mut self, branching: Branching, approx:ApproximateMethod, compile_timeout: u64) {
        let distributions: Vec<Vec<Rational>> = self.get_softmaxed_array().iter().map(|d| d.iter().map(|f| f.clone()).collect::<Vec<Rational>>()).collect();
        let mut train_dacs = generate_dacs(&self.clauses, &distributions, branching, self.epsilon, approx, compile_timeout);
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
    pub fn compute_gradients(&mut self, gradient_loss: &[Rational]) {
        self.zero_grads();
        for query_id in 0..self.train.len() {
            self.train[query_id].zero_paths();
            // Iterate on all nodes from the DAC, top-down way
            for node in self.train[query_id].iter_rev() {

                let start = self.train[query_id][node].input_start();
                let end = start + self.train[query_id][node].number_inputs();
                let value = self.train[query_id][node].value().clone();
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
                            let mut val = rational(0.0);
                            if *self.train[query_id][child].value() != 0.0 {
                                val = path_val.clone() * &value / self.train[query_id][child].value();
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
                        let mut factor = path_val.clone() * &gradient_loss[query_id];
                        if self.train[query_id][node].is_product() {
                            factor *= &value;
                            factor /= self.get_probability(d, v);
                        }
                        // Compute the gradient contribution for the value used in the node 
                        // and all the other possible values of the distribution (derivative of the softmax)
                        let mut sum_other_w = rational(0.0);
                        let child_w = self.get_probability(d, v);
                        for params in (0..self.unsoftmaxed_distributions[d].len()).filter(|i| *i != v) {
                            let weight = self.get_probability(d, params);
                            self.gradients[d][params] -= factor.clone() * &weight * &child_w;
                            sum_other_w += &weight;
                        }
                        self.gradients[d][v] += factor * child_w * sum_other_w;
                    }
                }
            }
        }
    }

    /// Update the distributions with the computed gradients and the learning rate, following an SGD approach
    pub fn update_distributions(&mut self, learning_rate: f64) {
        let lr = rational(learning_rate);
        for (distribution, grad) in self.unsoftmaxed_distributions.iter_mut().zip(self.gradients.iter()) {
            for (value, grad) in distribution.iter_mut().zip(grad.iter()) {
                *value -= grad * &lr;
            }
        }
    }

    /// Training loop for the train dacs, using the given training parameters
    //fn train(&mut self, params: &LearnParameters, inputs: &Vec<PathBuf>, branching: Branching, approx:ApproximateMethod, compile_timeout: u64) {
    pub fn train(&mut self, params: &LearnParameters, branching: Branching, approx:ApproximateMethod) {
        let mut prev_loss = rational(1.0);
        let mut count_no_improve = 0;
        self.log.start();
        let start = Instant::now();
        let to = Duration::from_secs(params.learning_timeout());
        
        // Evaluate the test set before training, if it exists
        if self.test.len() != 0 {
            let predictions = self.test();
            let test_loss = predictions.iter().enumerate().map(|(i, prediction)| {
                params.loss().loss(prediction, self.test.expected(i))
            }).collect::<Vec<Rational>>();
            self.log.log_test(&test_loss, self.epsilon, &predictions);
        }

        // Training loop
        let mut train_loss = vec![rational(0.0); self.train.len()];
        let mut train_grad = vec![rational(0.0); self.train.len()];
        for e in 0..params.nepochs() {
            // Update the learning rate
            let learning_rate = params.lr() * params.lr_drop().powf(((1+e) as f64/ params.epoch_drop() as f64).floor());
            // Forward pass
            let predictions = self.evaluate();
            // Compute the loss and the gradients
            for i in 0..predictions.len() {
                train_loss[i] = params.loss().loss(&predictions[i], self.train.expected(i));
                train_grad[i] = params.loss().gradient(&predictions[i], self.train.expected(i));
                if params.e_weighted() && self.epsilon != 0.0 {
                    //train_loss[i] *= 1.0 - self.train[i].epsilon()/self.epsilon;
                    let l = self.train.len() as f64;
                    train_grad[i] *= rational((1.0 - self.train[i].epsilon()/self.epsilon) * l / (l - 1.0));
                }
            }
            let avg_loss = train_loss.iter().sum::<Rational>() / rational(train_loss.len());
            self.compute_gradients(&train_grad);
            // Update the parameters
            self.update_distributions(learning_rate);
            // Log the epoch
            self.log.log_epoch(&train_loss, learning_rate, self.epsilon, &predictions);
            // Early stopping
            if do_early_stopping(&avg_loss, &prev_loss, &mut count_no_improve, params.early_stop_threshold(), params.patience(), params.early_stop_delta()) {
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
                self.recompile_dacs(branching, approx, params.compilation_timeout());
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
            let test_loss = predictions.iter().enumerate().map(|(i, prediction)| {
                params.loss().loss(prediction, self.test.expected(i))
            }).collect::<Vec<Rational>>();
            self.log.log_test(&test_loss, self.epsilon, &predictions);
        }
    }
}

impl <const S: bool> std::ops::Index<DacIndex> for Learner<S> {
    type Output = Dac;

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
pub fn softmax(x: &[Rational]) -> Vec<Rational> {
    let sum_exp: Rational = x.iter().map(|r| r.pow(10i64)).sum();
    x.iter().map(|r| r.pow(10i64) / &sum_exp).collect()
}

/// Generates a vector of optional Dacs from a list of input files
pub fn generate_dacs(queries_clauses: &Vec<Vec<Vec<isize>>>, distributions: &[Vec<Rational>],branching: Branching, epsilon: f64, approx: ApproximateMethod, timeout: u64) -> Vec<Dac> {
    queries_clauses.par_iter().map(|clauses| {
        // We compile the input. This can either be a .cnf file or a fdac file.
        // If the file is a fdac file, then we read directly from it
        let mut state = StateManager::default();
        let problem = create_problem(distributions, clauses, &mut state); //parser.problem_from_file(&mut state);
        let parameters = SolverParameters::new(u64::MAX, epsilon, timeout);
        let propagator = Propagator::new(&mut state);
        let component_extractor = ComponentExtractor::new(&problem, &mut state);
        let compiler = generic_solver(problem, state, component_extractor, branching, propagator, parameters, false, true);
        match approx {
            ApproximateMethod::Bounds => {
                match compiler {
                    crate::GenericSolver::Compiler(mut s) => s.compile(false),
                    crate::GenericSolver::LogCompiler(mut s) => s.compile(false),
                    _ => panic!("Search solver used for learning"),
                }
            },
            ApproximateMethod::LDS => {
                match compiler {
                    crate::GenericSolver::Compiler(mut s) => s.compile(true),
                    crate::GenericSolver::LogCompiler(mut s) => s.compile(true),
                    _ => panic!("Search solver used for learning"),
                }
            },
            
        }
    }).collect::<Vec<_>>()
}

/// Decides whether early stopping should be performed or not
pub fn do_early_stopping(avg_loss: &Rational, prev_loss: &Rational, count: &mut usize, stopping_criterion: &Rational, patience: usize, delta: f64) -> bool {
    if (avg_loss.clone() - prev_loss).abs() < delta {
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
pub struct Dataset {
    queries: Vec<Dac>,
    expected: Vec<Rational>,
}

impl Dataset {

    /// Creates a new dataset from the provided queries and expected probabilities
    pub fn new(queries: Vec<Dac>, expected: Vec<Rational>) -> Self {
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
    pub fn get_queries(&self) -> &Vec<Dac> {
        &self.queries
    }

    /// Returns a mutable reference to the queries of the dataset
    pub fn get_queries_mut(&mut self) -> &mut Vec<Dac> {
        &mut self.queries
    }

    /// Adds a query to the dataset
    pub fn add_query(&mut self, query: Dac, expected: f64) {
        self.queries.push(query);
        self.expected.push(rational(expected));
    }

    /// Returns the expected output for the required query
    pub fn expected(&self, query_index: usize) -> &Rational {
        &self.expected[query_index]
    }

    /// Sets the queries of the dataset
    pub fn set_queries(&mut self, queries: Vec<Dac>) {
        self.queries = queries;
    }
}

impl std::ops::Index<usize> for Dataset {
    type Output = Dac;

    fn index(&self, index: usize) -> &Self::Output {
        &self.queries[index]
    }
}

impl std::ops::IndexMut<usize> for Dataset {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.queries[index]
    }
}
