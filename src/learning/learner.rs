use crate::core::graph::*;
use rand::distributions;
use rug::{Assign, Float};
use crate::common::*;
use crate::learning::circuit::*;


pub struct Learner {
    dacs: Vec<Dac>,
    unsoftmaxed_distributions: Vec<Vec<f64>>,
    gradients: Vec<Vec<Float>>,
    lr: f64,
}

impl Learner {

    /// Creates a new learner from the given graphs.
    pub fn new(distributions:Vec<Vec<f64>>, lr:f64) -> Self {
        //let mut distributions: Vec<Vec<f64>> = vec![];
        let mut grads: Vec<Vec<Float>> = vec![];
        // let mut softmaxed_distributions: Vec<Vec<f64>> = vec![];
        /* for distribution in graph.distributions_iter() {
            let mut probabilities: Vec<f64>= graphs[0].distribution_variable_iter(distribution).map(|v| graphs[0].get_variable_weight(v).unwrap()).collect();
            for el in &mut probabilities {
                *el = el.log10();
            }
            let prob_len = probabilities.len();
            softmaxed_distributions.push(softmax(&probabilities));
            distributions.push(probabilities);
            grads.push(vec![f128!(0.0); prob_len]);
        } */
        /* for dac in dacs.iter_mut(){
            dac.push(Dac::new(&softmaxed_distributions));
        } */
        Self { 
            dacs: vec![], 
            unsoftmaxed_distributions: distributions, 
            gradients: grads,
            lr: lr }
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
    pub fn compute_gradients(&mut self, gradient_loss: f64){
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
                            let mut factor = path_val.clone() * gradient_loss;
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

}