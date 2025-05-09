use std::u64;

use pyo3::prelude::*;
use super::*;
use schlandals::learning::LearnParameters;
use schlandals::learning::learner::Learner;
use schlandals::learning::LossFunctions;
use schlandals::common::Semiring;
use schlandals::common::LearningMethod;

#[pyclass(name = "PyLearner")]
struct PyLearner {
    learner: Learner<false>,
    params: LearnParameters,
}

#[pymethods]
impl PyLearner {
    #[new]
    pub fn new(input: PathBuf, params: PyLearnParameters, branching: Option<PyBranching>, 
               outfolder: Option<PathBuf>, epsilon: Option<f64>, jobs: Option<usize>, semiring: Option<PySemiring>,
               trainfile: PathBuf, testfile: Option<PathBuf>, learning_m:PyLearningMethod,approx:PyApproximateMethod) -> Self {
        let sr = train::get_semiring_from_pysemiring(if let Some(v) = semiring { v } else { PySemiring::Probability });
        let br = get_branching_from_pybranching(if let Some(v) = branching { v } else { PyBranching::MinInDegree });
        let ap = get_approximatem_from_pyapproximatem(approx);
        let eps = if let Some(v) = epsilon { v } else { 0.0 };
        let jbs = if let Some(v) = jobs { v } else { 1 };
        let learner = match sr {
            Semiring::Probability => {
                Learner::<false>::new(input.clone(), schlandals::Args {input, evidence:None, timeout:params.compilation_timeout(), branching:br, statistics: true, 
                                    memory: u64::MAX, epsilon: eps, approx: ap, 
                                    subcommand: Some(schlandals::Command::Learn { trainfile, testfile, outfolder, lr:params.lr(), nepochs:params.nepochs(), 
                                            do_log: true, ltimeout: params.learning_timeout(), loss: train::get_loss_from_pyloss(params.loss()), jobs: jbs, 
                                            semiring: sr, optimizer: train::get_optimizer_from_pyoptimizer(params.optimizer()), lr_drop: params.lr_drop(), 
                                            epoch_drop: params.epoch_drop(), early_stop_threshold: params.early_stop_threshold(), early_stop_delta: params.early_stop_delta(),
                                            patience: params.patience(), equal_init:params.equal_init(), recompile:params.recompile(), e_weighted:params.e_weighted(), 
                                            learning_m:train::get_learningm_from_pylearningm(learning_m), lds_opti:false,})})
            }
        };
        Self{
            learner,
            params: get_param_from_pyparam(params),
        }
    }

    pub fn set_distributions(&mut self, distributions: Vec<Vec<f64>>) {
        let mut log_dist = vec![];
        for dist in distributions {
            log_dist.push(dist.iter().map(|x| x.ln()).collect());
        }
        self.learner.set_distributions(log_dist, false);
    }        

    pub fn get_distributions(&self) -> Vec<Vec<f64>> {
        self.learner.get_softmaxed_array().iter().map(|row| row.iter().map(|x| x.to_f64()).collect()).collect()
    }

    pub fn get_expected(&self) -> Vec<f64> {
        self.learner.get_train().get_expecteds().to_vec()
    }

    pub fn get_test_expected(&self) -> Vec<f64> {
        self.learner.get_test().get_expecteds().to_vec()
    }

    pub fn get_epsilon(&self, idx: usize) -> f64 {
        self.learner.get_train().get_queries()[idx].epsilon()
    }

    /* pub fn get_queries(&self) -> Vec<(f64,f64)> {
        self.learner.get_queries().iter().map(|x| (x.0, x.1)).collect()
    } */

    pub fn evaluate(&mut self, idxs: Option<Vec<usize>>) -> Vec<f64> {
        let mut eval = self.learner.evaluate();
        if let Some(v) = idxs {
            eval = v.iter().map(|i| eval[*i]).collect();
        }
        eval
    }

    pub fn evaluate_dist(&mut self, idxs: Option<Vec<usize>>, distributions: Option<Vec<Vec<Vec<f64>>>>) -> Vec<f64> {
        if let Some(ids) = idxs {
            if let Some(dists) = distributions {
                let mut res = vec![];
                for i in 0..dists.len() {
                    self.learner.set_distributions(dists[i].clone(), true);
                    res.push(self.learner.evaluate()[ids[i]]);
                }
                return res;
            }
            else {
                let eval = self.learner.evaluate();
                return ids.iter().map(|i| eval[*i]).collect();
            }
        }
        else {
            return self.learner.evaluate();
        }
    }

    pub fn update_distributions(&mut self, lr: f64) {
        self.learner.update_distributions(lr);
    }

    pub fn loss_and_grad(&mut self, predictions: Vec<f64>, expected: Vec<f64>, mut idxs:Option<Vec<usize>>) -> (Vec<f64>, Vec<Vec<f64>>) {
        // predictions and expected should have the same length, and correspond to the idxs in the same order, basically the batch considered
        let mut train_loss = vec![0.0; self.learner.get_train().len()];
        let mut train_grad = vec![0.0; self.learner.get_train().len()];

        if idxs.is_none(){
            idxs = Some((0..expected.len()).collect());
        }
        let idxs = idxs.unwrap();
        for (i, id) in idxs.iter().enumerate() {
            train_loss[*id] = self.params.loss().loss(predictions[i], expected[i]);
            train_grad[*id] = self.params.loss().gradient(predictions[i], expected[i]);
        }
        self.learner.compute_gradients(&train_grad, None);
        // Todo renvoyer seulement la loss des idx concernes et ?? pour le grad
        (idxs.iter().map(|i| train_loss[*i]).collect(), self.learner.get_gradients().iter().map(|row| row.iter().map(|x| x.to_f64()).collect()).collect())
    }

    pub fn eval_loss_grad(&mut self, idxs: Option<Vec<usize>>, distributions: Option<Vec<Vec<Vec<f64>>>>, expected:Vec<f64>) -> (Vec<f64>, Vec<f64>, Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        let mut preds = vec![];
        let mut loss = vec![];
        let mut grad = vec![];
        let mut agree = vec![];
        if let Some(ids) = idxs {
            if let Some(dists) = distributions {
                for i in 0..dists.len() {
                    self.learner.set_distributions(dists[i].clone(), true);
                    preds.push(self.learner.evaluate()[ids[i]]);
                    //println!("c size {}" , self.learner.get_train().get_queries()[ids[i]].number_nodes());
                    //println!("c\n{}", self.learner.get_train().get_queries()[ids[i]].as_graphviz());
                    loss.push(self.params.loss().loss(preds[i], expected[i]));
                    let mut grad_l = vec![0.0; self.learner.get_train().len()];
                    grad_l[ids[i]] = self.params.loss().gradient(preds[i], expected[i]);
                    agree.push(self.learner.compute_gradients(&grad_l, Some(vec![ids[i]]))[0].clone());
                    grad.push(self.learner.get_gradients().iter().map(|row| row.iter().map(|x| x.to_f64()).collect()).collect());
                }
            }
            else {
                /* preds = ids.iter().map(|i| self.learner.evaluate()[*i]).collect();
                loss = ids.iter().zip(preds.iter()).map(|(i, p)| self.params.loss().loss(*p, expected[*i])).collect();
                let grad_l = ids.iter().zip(preds.iter()).map(|(i, p)| self.params.loss().gradient(*p, expected[*i])).collect();
                for i in 0..grad_l.len() {
                    self.learner.compute_gradients(&grad_l[i], Some(vec![ids[i]]));
                    grad.push(self.learner.get_gradients().iter().map(|row| row.iter().map(|x| x.to_f64()).collect()).collect());
                } */
                for i in 0..ids.len() {
                    preds.push(self.learner.evaluate()[ids[i]]);
                    loss.push(self.params.loss().loss(preds[i], expected[i]));
                    let mut grad_l = vec![0.0; self.learner.get_train().len()];
                    grad_l[ids[i]] = self.params.loss().gradient(preds[i], expected[i]);
                    agree.push(self.learner.compute_gradients(&grad_l, Some(vec![ids[i]]))[0].clone());
                    grad.push(self.learner.get_gradients().iter().map(|row| row.iter().map(|x| x.to_f64()).collect()).collect());
                }
            }
        }
        else {
            if let Some(dists) = distributions {
                for i in 0..dists.len() {
                    self.learner.set_distributions(dists[i].clone(), true);
                    preds.push(self.learner.evaluate()[i]);
                    loss.push(self.params.loss().loss(preds[i], expected[i]));
                    let mut grad_l = vec![0.0; self.learner.get_train().len()];
                    grad_l[i] = self.params.loss().gradient(preds[i], expected[i]);
                    agree.push(self.learner.compute_gradients(&grad_l, Some(vec![i]))[0].clone());
                    grad.push(self.learner.get_gradients().iter().map(|row| row.iter().map(|x| x.to_f64()).collect()).collect());
                }
            }
            else {
                preds = self.learner.evaluate();
                loss = expected.iter().zip(preds.iter()).map(|(e, p)| self.params.loss().loss(*p, *e)).collect();
                let grad_l: Vec<f64> = expected.iter().zip(preds.iter()).map(|(e, p)| self.params.loss().gradient(*p, *e)).collect();
                for i in 0..grad_l.len() {
                    agree.push(self.learner.compute_gradients(&grad_l, Some(vec![i]))[0].clone());
                    grad.push(self.learner.get_gradients().iter().map(|row| row.iter().map(|x| x.to_f64()).collect()).collect());
                }
            }
        }
        (preds, loss, grad, agree)    
    }


    pub fn test_eval_loss(&mut self, idxs: Option<Vec<usize>>, distributions: Option<Vec<Vec<Vec<f64>>>>, expected:Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        let mut preds = vec![];
        let mut loss = vec![];
        if let Some(ids) = idxs {
            if let Some(dists) = distributions {
                for i in 0..dists.len() {
                    self.learner.set_distributions(dists[i].clone(), true);
                    preds.push(self.learner.test()[ids[i]]);
                    //println!("c size {}" , self.learner.get_train().get_queries()[ids[i]].number_nodes());
                    //println!("c\n{}", self.learner.get_train().get_queries()[ids[i]].as_graphviz());
                    loss.push(self.params.loss().loss(preds[i], expected[i]));
                }
            }
            else {
                for i in 0..ids.len() {
                    preds.push(self.learner.test()[ids[i]]);
                    loss.push(self.params.loss().loss(preds[i], expected[i]));
                }
            }
        }
        else {
            if let Some(dists) = distributions {
                for i in 0..dists.len() {
                    self.learner.set_distributions(dists[i].clone(), true);
                    preds.push(self.learner.test()[i]);
                    loss.push(self.params.loss().loss(preds[i], expected[i]));
                }
            }
            else {
                preds = self.learner.test();
                loss = expected.iter().zip(preds.iter()).map(|(e, p)| self.params.loss().loss(*p, *e)).collect();
            }
        }
        (preds, loss)
    }

    pub fn as_graphviz(&self, idx: usize) -> String {
        self.learner.get_train().get_queries()[idx].as_graphviz()
    }
}




#[pymodule]
#[pyo3(name="learner")]
pub fn learner_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "learner")?;

    module.add_class::<PyLearner>()?;

    parent_module.add_submodule(module)?;
    py.import("sys")?.getattr("modules")?.set_item("pyschlandals.learner", module)?;
    Ok(())
}