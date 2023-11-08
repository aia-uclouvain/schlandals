use std::{fmt, path::PathBuf, fs,fs::{File}, io::{BufRead, BufReader}};

use std::io::{self, Write};
use rug::{Assign, Float};
use crate::common::*;
use crate::learning::circuit::*;
use rand::Rng;
use super::logger::Logger;
use search_trail::StateManager;
use crate::branching::*;
use crate::parser;
use crate::propagator::Propagator;
use crate::core::components::ComponentExtractor;
use crate::Branching;
use super::exact::DACCompiler;


pub struct Learner<const S: bool> {
    pub dacs: Vec<Dac>,
    unsoftmaxed_distributions: Vec<Vec<f64>>,
    gradients: Vec<Vec<Float>>,
    lr: f64,
    expected_distribution: Vec<Vec<f64>>,
    expected_outputs: Vec<f64>,
    log: Logger<S>,
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

impl <const S: bool> Learner<S> {

    /// Creates a new learner from the given graphs.
    pub fn new(inputs: Vec<PathBuf>, branching: Branching, timeout:u64, folderdac: Option<PathBuf>, read:bool) -> Self {
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
                unsoftmaxed_distributions: unsoftmaxed_distributions, 
                gradients: grads,
                lr: 0.0,
                expected_distribution: distributions,
                expected_outputs: vec![],
                log: Logger::default(),
            };

            for input in &inputs {
                println!("Compiling {}", input.display());
                let mut state = StateManager::default();
                let propagator = Propagator::new(&mut state);
                let graph = parser::graph_from_ppidimacs(&input, &mut state);
                let component_extractor = ComponentExtractor::new(&graph, &mut state);
                let mut branching_heuristic: Box<dyn BranchingDecision> = match branching {
                    Branching::MinInDegree => Box::<MinInDegree>::default(),
                    Branching::MinOutDegree => Box::<MinOutDegree>::default(),
                    Branching::MaxDegree => Box::<MaxDegree>::default(),
                    Branching::VSIDS => Box::<VSIDS>::default(),
                };
                let mut compiler = DACCompiler::new(graph, state, component_extractor, branching_heuristic.as_mut(), propagator);
                let res = compiler.compile(timeout);
                if let Some(dac) = res {
                    learner.add_dac(dac);
                }
                else {
                    println!("Skipped");
                }
            }
            let expected_outputs = learner.evaluate();
            learner.expected_outputs = expected_outputs;
            learner.unsoftmaxed_distributions = rand_init;

            if let Some(f) = folderdac {
                println!("{:?}", f.join("distributions.fdist"));
                let mut outfile = File::create(f.join("distributions.fdist")).unwrap();
                match outfile.write(format!("{}", learner).as_bytes()) {
                    Ok(_) => (),
                    Err(e) => println!("Could not write the distributions into the fdist file: {:?}", e),
                }
                for (i, dac) in learner.dacs.iter().enumerate() {
                    let mut outfile = File::create(f.join(format!("{}.fdac", i))).unwrap();
                    match outfile.write(format!("{}", dac).as_bytes()) {
                        Ok(_) => (),
                        Err(e) => println!("Could not write the circuit into the fdac file: {:?}", e),
                    }
                }
            }
            learner

        } else {
            println!("Reading the distributions from the given folder");
            if let Some(f) = folderdac{
                let mut learner = Self::from_file(&f.join("distributions.fdist"));
                let paths = fs::read_dir(f).unwrap();
                for path in paths {
                    let path = path.unwrap().path();
                    if path.is_file() && path.extension().unwrap() == "fdac" {
                        learner.add_dac(Dac::from_file(&path));
                        let file = File::open(path).unwrap();
                        let reader = BufReader::new(file);
                        for line in reader.lines() {
                            let l = line.unwrap();
                            let split = l.split_whitespace().collect::<Vec<&str>>();
                            if l.starts_with("evaluate"){
                                learner.expected_outputs.push(split[1].parse::<f64>().unwrap());
                            }
                        }
                    }
                }
                println!("grads: {:?}", learner.gradients);
                learner
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

    pub fn add_dac(&mut self, dac: Dac) {
        self.dacs.push(dac);
    }

    // --- Evaluation --- //

    fn reset_dacs(&mut self) {
        for dac_i in 0..self.dacs.len() {
            for node_i in 0..self.dacs[dac_i].nodes.len() {
                match self.dacs[dac_i].nodes[node_i].get_type() {
                    TypeNode::Sum => {
                        self.dacs[dac_i].nodes[node_i].set_value(0.0);
                    },
                    TypeNode::Product => {
                        self.dacs[dac_i].nodes[node_i].set_value(1.0);
                    },
                    TypeNode::Distribution{d,v} => {
                        let proba = self.get_probability(d, v);
                        self.dacs[dac_i].nodes[node_i].set_value(proba);
                    },
                }
                self.dacs[dac_i].nodes[node_i].set_path_value(f128!(1.0));
            }
        }
    }

    // Evaluate the different DACs and return the results
    pub fn evaluate(&mut self) -> Vec<f64> {
        self.reset_dacs();
        let mut evals: Vec<f64> = vec![];
        for dac in self.dacs.iter_mut() {
            evals.push(dac.evaluate().to_f64());
        }
        evals
    }

    // --- Gradient computation --- //

    // Compute the gradient of the distributions, from the different DAC queries
    pub fn compute_gradients(&mut self, gradient_loss: Vec<f64>){
        self.zero_grads();
        for dac_i in 0..self.dacs.len() {
            // Iterate on the different DAC queries
            for node in (0..self.dacs[dac_i].nodes.len()).map(NodeIndex).rev(){
                // Iterate on all nodes from the DAC, top-down way
                let start = self.dacs[dac_i].nodes[node.0].get_input_start();
                let end = start + self.dacs[dac_i].nodes[node.0].get_number_inputs();
                let value = self.dacs[dac_i].nodes[node.0].get_value();
                let path_val = self.dacs[dac_i].nodes[node.0].get_path_value();
                // Update the path value for the children sum, product nodes 
                // and compute the gradient for the children leaf distributions
                for i in start..end {
                    let child = self.dacs[dac_i].get_input_at(i);
                    match self.dacs[dac_i].nodes[child.0].get_type() {
                        TypeNode::Sum => {
                            let mut val = path_val.clone() * &value;
                            val /= self.dacs[dac_i].nodes[child.0].get_value();
                            self.dacs[dac_i].nodes[child.0].set_path_value(val);
                        },
                        TypeNode::Product => {
                            self.dacs[dac_i].nodes[child.0].set_path_value(path_val.clone());
                        },
                        TypeNode::Distribution{d,v} => {
                            // Compute the gradient for children that are leaf distributions
                            let mut factor = path_val.clone() * gradient_loss[dac_i];
                            if matches!(self.dacs[dac_i].nodes[node.0].get_type(), TypeNode::Product) {
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
            let mut loss = 0.0;
            let mut loss_grad = vec![0.0; self.dacs.len()];
            for dac_i in 0..self.dacs.len() {
                loss += (predictions[dac_i] - self.expected_outputs[dac_i]).powi(2);
                loss_grad[dac_i] = 2.0 * (predictions[dac_i] - self.expected_outputs[dac_i]);
            }
            loss /= self.dacs.len() as f64;
            self.compute_gradients(loss_grad);
            self.update_distributions();
            self.log.add_epoch(loss, &self.expected_distribution, &self.get_softmaxed_array(), self.lr);
        }

        self.training_to_file(fout);
    }
    /*  else if l.starts_with('d') {
                let dom_size = split[1].parse::<usize>().unwrap();
                let mut probabilities = split[2..(2+dom_size)].iter().map(|s| s.parse::<f64>().unwrap()).collect::<Vec<f64>>();
                let mut outputs: Vec<(CircuitNodeIndex, usize)> = vec![];
                for i in ((2+dom_size)..split.len()).step_by(2) {
                    let output_node = CircuitNodeIndex(split[i].parse::<usize>().unwrap());
                    let value = split[i+1].parse::<usize>().unwrap();
                    outputs.push((output_node, value));
                    input_distributions_node[output_node.0].push((DistributionIndex(dac.distribution_nodes.len()), value));
                }
                for el in &mut probabilities {
                    *el = el.log(std::f64::consts::E);
                }
                let prob_len = probabilities.len();
                dac.distribution_nodes.push(DistributionNode {
                    unsoftmaxed_probabilities: probabilities,
                    outputs,
                    grad_value: vec![f128!(0.0); prob_len],
                });
            } */
}
impl <const S: bool> fmt::Display for Learner<S> {
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

impl <const S: bool> Learner<S> {
    pub fn from_file(filepath: &PathBuf) -> Self {
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
        }
    }
}