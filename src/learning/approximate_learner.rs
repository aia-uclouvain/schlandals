use std::{fmt, path::PathBuf, fs,fs::File, io::{BufRead, BufReader}};

use std::io::{self, Write};
use std::cmp::min;
use rug::{Assign, Float};
use crate::common::*;
use crate::learning::circuit::*;
use rand::Rng;
use super::logger::Logger;
use search_trail::StateManager;
use crate::heuristics::BranchingDecision;
use crate::heuristics::branching_exact::*;
use crate::heuristics::branching_limited::*;
use crate::parser;
use crate::propagator::Propagator;
use crate::core::components::ComponentExtractor;
use crate::Branching;
use super::exact::DACCompiler;
use crate::solvers::{QuietSearchSolver, StatSearchSolver};
use sysinfo::{SystemExt, System};

pub struct ApproximateLearner<const S: bool> {
    pub dacs: Vec<Vec<Dac>>, // [nb_approx, nb_dac]
    unsoftmaxed_distributions: Vec<Vec<f64>>,
    gradients: Vec<Vec<Float>>,
    lr: f64,
    epsilon: f64,
    expected_distribution: Vec<Vec<f64>>,
    expected_outputs: Vec<f64>,
    log: Logger<S>,
    nb_approx: usize,
    branching: Branching,
    inputs: Vec<PathBuf>,
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

impl <const S: bool> ApproximateLearner<S> {

    /// Creates a new learner from the given graphs.
    pub fn new(inputs: Vec<PathBuf>, epsilon:f64, branching: Branching, timeout:u64, folderdac: Option<PathBuf>, read:bool, nb_approx:usize) -> Self {
        if !read {
            let mut distributions: Vec<Vec<f64>> = vec![];
            let mut grads: Vec<Vec<Float>> = vec![];
            let mut unsoftmaxed_distributions: Vec<Vec<f64>> = vec![];

            let mut rng = rand::thread_rng();
            let mut rand_init: Vec<Vec<f64>> = vec![];

            let mut state = StateManager::default();
            let graph = parser::graph_from_ppidimacs(&inputs[0], &mut state);
            for (i, distribution) in graph.distributions_iter().enumerate() {
                let probabilities: Vec<f64>= graph[distribution].iter_variables().map(|v| graph[v].weight().unwrap()).collect();
                distributions.push(probabilities);

                let mut vector: Vec<f64> = vec![0.0; distributions[i].len()];
                let mut unsoftmaxed_vector: Vec<f64> = vec![0.0; distributions[i].len()];
                for j in 0..distributions[i].len() {
                    vector[j] = rng.gen_range(0.0..1.0);
                    vector[j] = vector[j].log(std::f64::consts::E);
                    unsoftmaxed_vector[j] = distributions[i][j].log(std::f64::consts::E);
                }
                rand_init.push(vector);
                unsoftmaxed_distributions.push(unsoftmaxed_vector);
                grads.push(vec![f128!(0.0); distributions[i].len()]);
            }

            let mut learner = Self { 
                dacs: vec![], 
                unsoftmaxed_distributions, 
                gradients: grads,
                lr: 0.0,
                epsilon,
                expected_distribution: distributions,
                expected_outputs: vec![],
                log: Logger::default(),
                nb_approx,
                branching,
                inputs: vec![],
            };
            let distributions_len = learner.expected_distribution.len();

            let mut paths: Vec<PathBuf> = vec![];
            for i in (0..distributions_len).step_by(distributions_len/nb_approx){
                let mut batch: Vec<Dac> = vec![];
               for input in &inputs {
                    println!("Compiling {}", input.display());
                    let mut state = StateManager::default();
                    let propagator = Propagator::new(&mut state);
                    let graph = parser::graph_from_ppidimacs(&input, &mut state);
                    let component_extractor = ComponentExtractor::new(&graph, &mut state);
                    let mut branching_heuristic = Box::from(Counting::new(i, min(distributions_len/nb_approx,distributions_len-i)));
                    let mut compiler = DACCompiler::new(graph, state, component_extractor, branching_heuristic.as_mut(), propagator);
                    let res = compiler.compile(timeout);
                    if let Some(dac) = res {
                        batch.push(dac);
                        if i==0 {paths.push(input.clone());}
                    }
                    else {
                        println!("Skipped");
                    }
                } 
                learner.add_dacs(batch);
            }
            
            let expected_outputs = learner.evaluate();
            learner.expected_outputs = expected_outputs[0].clone();
            learner.unsoftmaxed_distributions = rand_init;
            learner.inputs = paths.clone();

            if let Some(f) = folderdac {
                Self::to_folder(&learner, paths, f);
            }
            learner

        } else {
            println!("Reading the distributions from the given folder");
            if let Some(f) = folderdac{
                Self::from_folder(f, branching, nb_approx, epsilon, inputs)
            }
            else {
                panic!("No folder given to read the distributions from");
            }  
        }
    }

    // --- Getters --- //
    fn get_softmaxed(&self, distribution: usize) -> Vec<f64> {
        softmax(&self.unsoftmaxed_distributions[distribution])
    }

    pub fn get_probability(&self, distribution: usize, index: usize) -> f64 {
        self.get_softmaxed(distribution)[index]
    }

    fn get_softmaxed_array(&self) -> Vec<Vec<f64>> {
        let mut softmaxed: Vec<Vec<f64>> = vec![];
        for distribution in self.unsoftmaxed_distributions.iter() {
            softmaxed.push(softmax(distribution));
        }
        softmaxed
    }

    // --- Setters --- //
    pub fn zero_grads(&mut self) {
        for grad in self.gradients.iter_mut() {
            for el in grad.iter_mut() {
                el.assign(0.0);
            }
        }
    }

    pub fn add_dacs(&mut self, dacs: Vec<Dac>) {
        self.dacs.push(dacs);
    }

    // --- Evaluation --- //

    fn reset_dacs(&mut self) {
        for batch_i in 0..self.dacs.len() {
            for dac_i in 0..self.dacs[batch_i].len() {
                for node_i in 0..self.dacs[batch_i][dac_i].nodes.len() {
                    match self.dacs[batch_i][dac_i].nodes[node_i].get_type() {
                        TypeNode::Sum => {
                            if self.dacs[batch_i][dac_i].nodes[node_i].get_propagation().is_empty() {
                                self.dacs[batch_i][dac_i].nodes[node_i].set_value(0.0);
                            }
                            else {
                                self.reset_approx_node(batch_i, dac_i, node_i);
                            }
                        },
                        TypeNode::Product => {
                            if self.dacs[batch_i][dac_i].nodes[node_i].get_propagation().is_empty() {
                                self.dacs[batch_i][dac_i].nodes[node_i].set_value(1.0);
                            }
                            else {
                                self.reset_approx_node(batch_i, dac_i, node_i);
                            }
                        },
                        TypeNode::Distribution{d,v} => {
                            let proba = self.get_probability(d, v);
                            self.dacs[batch_i][dac_i].nodes[node_i].set_value(proba);
                        },
                    }
                    self.dacs[batch_i][dac_i].nodes[node_i].set_path_value(f128!(1.0));
                }
            }
        }
    }

    fn reset_approx_node(&mut self, batch_i:usize, dac_i:usize, node_i:usize){
        let mut state = StateManager::default();
        let propagator = Propagator::new(&mut state);
        let graph = parser::graph_from_ppidimacs(&self.inputs[dac_i], &mut state);
        let component_extractor = ComponentExtractor::new(&graph, &mut state);
        let mut branching_heuristic: Box<dyn BranchingDecision> = match self.branching {
            Branching::MinInDegree => Box::<MinInDegree>::default(),
            Branching::MinOutDegree => Box::<MinOutDegree>::default(),
            Branching::MaxDegree => Box::<MaxDegree>::default(),
        };
        let sys = System::new_all();
        let mlimit = sys.total_memory() / 1000000;
        let mut solver = QuietSearchSolver::new(
            graph,
            state,
            component_extractor,
            branching_heuristic.as_mut(),
            propagator,
            mlimit,
            self.epsilon,
        );
        solver.add_to_propagation_stack(self.dacs[batch_i][dac_i].nodes[node_i].get_propagation().clone());
        self.dacs[batch_i][dac_i].nodes[node_i].set_value(solver.solve().unwrap().to_f64());
    }

    // Evaluate the different DACs and return the results
    pub fn evaluate(&mut self) -> Vec<Vec<f64>> {
        self.reset_dacs();
        let mut evals: Vec<Vec<f64>> = vec![];
        for batch in self.dacs.iter_mut() {
            let mut evals_batch: Vec<f64> = vec![];
            for dac in batch.iter_mut() {
                evals_batch.push(dac.evaluate().to_f64());
            }
            evals.push(evals_batch);
        }
        evals
    }

    // --- Gradient computation --- //

    // Compute the gradient of the distributions, from the different DAC queries
    pub fn compute_gradients(&mut self, batch_i:usize, gradient_loss: Vec<f64>){
        self.zero_grads();
        for dac_i in 0..self.dacs[batch_i].len() {
            // Iterate on the different DAC queries
            for node in (0..self.dacs[batch_i][dac_i].nodes.len()).map(NodeIndex).rev(){
                // Iterate on all nodes from the DAC, top-down way
                let start = self.dacs[batch_i][dac_i].nodes[node.0].get_input_start();
                let end = start + self.dacs[batch_i][dac_i].nodes[node.0].get_number_inputs();
                let value = self.dacs[batch_i][dac_i].nodes[node.0].get_value();
                let path_val = self.dacs[batch_i][dac_i].nodes[node.0].get_path_value();
                // Update the path value for the children sum, product nodes 
                // and compute the gradient for the children leaf distributions
                for i in start..end {
                    let child = self.dacs[batch_i][dac_i].get_input_at(i);
                    match self.dacs[batch_i][dac_i].nodes[child.0].get_type() {
                        TypeNode::Sum => {
                            let mut val = path_val.clone() * &value;
                            val /= self.dacs[batch_i][dac_i].nodes[child.0].get_value();
                            self.dacs[batch_i][dac_i].nodes[child.0].set_path_value(val);
                        },
                        TypeNode::Product => {
                            self.dacs[batch_i][dac_i].nodes[child.0].set_path_value(path_val.clone());
                        },
                        TypeNode::Distribution{d,v} => {
                            // Compute the gradient for children that are leaf distributions
                            let mut factor = path_val.clone() * gradient_loss[dac_i];
                            if matches!(self.dacs[batch_i][dac_i].nodes[node.0].get_type(), TypeNode::Product) {
                                factor *= &value;
                                factor /= self.get_probability(d, v);
                            }
                            
                            // Compute the gradient contribution for the value used in the node and all the other possible values of the distribution
                            let mut sum_other_w = f128!(0.0);
                            let child_w = self.get_probability(d, v);
                            for params in 0..self.unsoftmaxed_distributions[d].len() {
                                let weight = self.get_probability(d, params);
                                if params != v {
                                    // For the other possible values of the distribution, the gradient contribution
                                    // is simply the dactor and the product of both weights
                                    self.gradients[d][params] -= factor.clone() * weight.clone() * child_w.clone();
                                    sum_other_w += weight.clone();
                                }
                            }
                            self.gradients[d][v] += factor.clone() * child_w.clone() * sum_other_w.clone();
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
    fn training_to_file(& self, fout: Option<PathBuf>) {
        let mut out_writer = match &fout {
            Some(x) => {
                Box::new(File::create(&x).unwrap()) as Box<dyn Write>
            }
            None => Box::new(io::stdout()) as Box<dyn Write>,
        };
        writeln!(out_writer, "Obtained distributions:").unwrap();
        for i in 0..self.unsoftmaxed_distributions.len() {
            writeln!(out_writer, "Distribution {}: {:?}", i, self.get_softmaxed(i)).unwrap();
        }

        let mut csv_file = match &fout {
            Some(x) => {
                Box::new(File::create(x.with_extension("csv")).unwrap()) as Box<dyn Write>
            }
            None => Box::new(io::stdout()) as Box<dyn Write>,
        };
        writeln!(csv_file, "{}", self.log).unwrap();
        
    }

    pub fn train(&mut self, nepochs:usize, lr:f64, fout: Option<PathBuf>) {
        self.lr = lr;
        self.log.start();

        for e in 0..nepochs {
            let do_print = e % 500 == 0;
            let predictions = self.evaluate();
            if do_print { println!("--- Epoch {} ---\n Predictions: {:?} \nExpected: {:?}\n", e, predictions, self.expected_outputs);}
            let mut total_loss = vec![0.0; self.nb_approx];
            for batch_i in 0..self.dacs.len() {
                let mut loss = vec![0.0; self.dacs[batch_i].len()];
                let mut loss_grad = vec![0.0; self.dacs[batch_i].len()];
                for dac_i in 0..self.dacs[batch_i].len() {
                    loss[dac_i] = (predictions[batch_i][dac_i] - self.expected_outputs[dac_i]).powi(2);
                    loss_grad[dac_i] = 2.0 * (predictions[batch_i][dac_i] - self.expected_outputs[dac_i]);
                    total_loss[dac_i] += loss[dac_i]/self.nb_approx as f64;
                }
                self.compute_gradients(batch_i, loss_grad);
                self.update_distributions();
            }
            self.log.add_epoch(total_loss, &self.expected_distribution, &self.get_softmaxed_array(), &self.gradients, self.lr);
        }

        self.training_to_file(fout);
    }
}

impl <const S: bool> fmt::Display for ApproximateLearner<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.expected_distribution.len() {
            write!(f, "d {}", self.expected_distribution[i].len())?;
            for p_i in 0..self.expected_distribution[i].len(){
                write!(f, " {:.5}", self.expected_distribution[i][p_i])?;
            }
            writeln!(f)?;
        }
        fmt::Result::Ok(())
    }
}

impl <const S: bool> ApproximateLearner<S> {
    pub fn to_folder(learner: &ApproximateLearner<S>, paths: Vec<PathBuf>, f: PathBuf) {
        println!("{:?}", f.join("distributions.fdist"));
        let mut outfile = File::create(f.join("distributions.fdist")).unwrap();
        match outfile.write(format!("{}", learner).as_bytes()) {
            Ok(_) => (),
            Err(e) => println!("Could not write the distributions into the fdist file: {:?}", e),
        }
        for i in 0..learner.dacs.len() {
            for (j, dac) in learner.dacs[i].iter().enumerate() {
                let mut outfile = File::create(f.join(format!("{}_{}.fdac", i,j))).unwrap();
                match outfile.write(format!("{}\n{}", dac, paths[j].display()).as_bytes()) {
                    Ok(_) => (),
                    Err(e) => println!("Could not write the circuit into the fdac file: {:?}", e),
                }
            }
        }
    }

    pub fn from_folder(f:PathBuf, branching: Branching, nb_approx: usize, epsilon:f64, inputs: Vec<PathBuf> ) -> ApproximateLearner<S> {
        let mut learner = Self::from_file(&f.join("distributions.fdist"), branching, nb_approx, epsilon, inputs);
        let paths = fs::read_dir(f).unwrap();
        let mut all_dacs: Vec<Vec<Dac>> = vec![];
        for path in paths {
            let path = path.unwrap().path();
            if path.is_file() && path.extension().unwrap() == "fdac" {
                let mut batch_i: usize = all_dacs.len();
                let mut dac_i : usize = all_dacs[batch_i].len();
                if let Some(file_name) = path.file_stem() {
                    if let Some(file_str) = file_name.to_str() {
                        let parts: Vec<&str> = file_str.split('_').collect();
                        if let Ok(x) = parts[0].parse::<usize>() {
                            if let Ok(y) = parts[1].parse::<usize>() {
                                batch_i = x;
                                dac_i = y;
                            }
                        }
                    }
                }
                if all_dacs.len() <= batch_i {
                    all_dacs.resize_with(batch_i, ||vec![]);
                }
                if all_dacs[batch_i].len() <= dac_i {
                    all_dacs[batch_i].resize_with(dac_i, ||Dac::new());
                }
                if learner.inputs.len() <= dac_i {
                    learner.inputs.resize_with(dac_i, ||PathBuf::new());
                }
                all_dacs[batch_i][dac_i] = Dac::from_file(&path);
                let file = File::open(path).unwrap();
                let reader = BufReader::new(file);
                for line in reader.lines() {
                    let l = line.unwrap();
                    let split = l.split_whitespace().collect::<Vec<&str>>();
                    if l.starts_with("evaluate"){
                        if learner.expected_outputs.len() <= dac_i {
                            learner.expected_outputs.resize_with(dac_i, ||0.0);
                        }
                        learner.expected_outputs[dac_i] = split[1].parse::<f64>().unwrap();
                    }
                    if l.ends_with(".cnf") && batch_i==0 {
                        learner.inputs[dac_i] = PathBuf::from(l);
                    }
                }
            }
        }
        learner
    }
}

impl <const S: bool> ApproximateLearner<S> {
    pub fn from_file(filepath: &PathBuf, branching:Branching, nb_approx:usize, epsilon:f64, inputs:Vec<PathBuf>) -> Self {
        let mut expected_probabilities: Vec<Vec<f64>> = vec![];
        let mut unsoftmaxed_probabilities: Vec<Vec<f64>> = vec![];
        let mut gradients: Vec<Vec<Float>> = vec![];

        let mut rand_init: Vec<Vec<f64>> = vec![];
        let mut rng = rand::thread_rng();

        let file = File::open(filepath).unwrap();
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let l = line.unwrap();
            let split = l.split_whitespace().collect::<Vec<&str>>();
            if l.starts_with('d') {
                let dom_size = split[1].parse::<usize>().unwrap();
                let probabilities = split[2..(2+dom_size)].iter().map(|s| s.parse::<f64>().unwrap()).collect::<Vec<f64>>();
                let mut unsoftmaxed_vector = probabilities.clone();
                let mut vector: Vec<f64> = vec![];
                for el in &mut unsoftmaxed_vector {
                    *el = el.log(std::f64::consts::E);
                    let rnd: f64 = rng.gen_range(0.0..1.0);
                    vector.push(rnd.log(std::f64::consts::E));
                }
                rand_init.push(vector);
                gradients.push(vec![f128!(0.0); dom_size]);
                unsoftmaxed_probabilities.push(unsoftmaxed_vector);
                expected_probabilities.push(probabilities);
            }
        }

        Self {
            dacs: vec![],
            unsoftmaxed_distributions: rand_init,
            gradients: gradients,
            lr: 0.0,
            expected_distribution: expected_probabilities,
            expected_outputs: vec![],
            log: Logger::default(),
            branching,
            nb_approx,
            epsilon,
            inputs,
        }
    }
}