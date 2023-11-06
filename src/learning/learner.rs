use std::path::PathBuf;

use std::io::{self, Write};
use std::fs::File;
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
    dacs: Vec<Dac>,
    unsoftmaxed_distributions: Vec<Vec<f64>>,
    gradients: Vec<Vec<Float>>,
    lr: f64,
    expected_distribution: Vec<Vec<f64>>,
    expected_outputs: Vec<f64>,
    log: Logger<S>,
}

impl <const S: bool> Learner<S> {

    /// Creates a new learner from the given graphs.
    pub fn new(distributions:Vec<Vec<f64>>, inputs: Vec<PathBuf>, branching: Branching, timeout:u64) -> Self {
        let mut grads: Vec<Vec<Float>> = vec![];
        let mut rng = rand::thread_rng();
        let mut rand_init: Vec<Vec<f64>> = vec![];
        for i in 0..distributions.len() {
            let mut vector: Vec<f64> = vec![0.0; distributions[i].len()];
            for j in 0..distributions[i].len() {
                vector[j] = rng.gen_range(0.0..1.0);
                vector[j] = vector[j].log10();
            }
            rand_init.push(vector);
            grads.push(vec![f128!(0.0); distributions[i].len()]);
        }

        let mut learner = Self { 
            dacs: vec![], 
            unsoftmaxed_distributions: rand_init, 
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
        learner
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

    pub fn add_dac(&mut self, mut dac: Dac) {
        self.expected_outputs.push(dac.evaluate().to_f64());
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
    pub fn evaluate(&mut self) -> Vec<Float> {
        self.reset_dacs();
        let mut evals: Vec<Float> = vec![];
        for dac in self.dacs.iter_mut() {
            evals.push(dac.evaluate());
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
                loss += (predictions[dac_i].to_f64() - self.expected_outputs[dac_i]).powi(2);
                loss_grad[dac_i] = 2.0 * (predictions[dac_i].to_f64() - self.expected_outputs[dac_i]);
            }
            loss /= self.dacs.len() as f64;
            self.compute_gradients(loss_grad);
            self.update_distributions();
            self.log.add_epoch(loss, &self.expected_distribution, &self.get_softmaxed_array(), self.lr);
        }

        self.training_to_file(fout);
    }

}