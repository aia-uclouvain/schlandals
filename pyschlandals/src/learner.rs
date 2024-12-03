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
    learner: Learner<true>,
    params: LearnParameters,
}

#[pymethods]
impl PyLearner {
    #[new]
    pub fn new(input: PathBuf, expected: Vec<f64>, params: PyLearnParameters, branching: Option<PyBranching>, 
               outfolder: Option<PathBuf>, epsilon: Option<f64>, jobs: Option<usize>, semiring: Option<PySemiring>,
               queries: Option<Vec<(f64,f64)>>, trainfile: PathBuf, testfile: Option<PathBuf>, learning_m:PyLearningMethod) -> Self {
        let sr = train::get_semiring_from_pysemiring(if let Some(v) = semiring { v } else { PySemiring::Probability });
        let br = get_branching_from_pybranching(if let Some(v) = branching { v } else { PyBranching::MinInDegree });
        let eps = if let Some(v) = epsilon { v } else { 0.0 };
        let jbs = if let Some(v) = jobs { v } else { 1 };
        let learner = match sr {
            Semiring::Probability => {
                /* Learner::<true>::new(inputs, expected, eps, br, outfolder, jbs, prm.compilation_timeout(), vec![], vec![], queries) */
                Learner::<true>::new(input.clone(), schlandals::Args {input, evidence:None, timeout:params.compilation_timeout(), branching:br, statistics: true, 
                                    memory: u64::MAX, epsilon: eps, approx: schlandals::common::ApproximateMethod::LDS, 
                                    subcommand: Some(schlandals::Command::Learn { trainfile, testfile, outfolder, lr:params.lr(), nepochs:params.nepochs(), 
                                            do_log: true, ltimeout: params.learning_timeout(), loss: train::get_loss_from_pyloss(params.loss()), jobs: jbs, 
                                            semiring: sr, optimizer: train::get_optimizer_from_pyoptimizer(params.optimizer()), lr_drop: params.lr_drop(), 
                                            epoch_drop: params.epoch_drop(), early_stop_threshold: params.early_stop_threshold(), early_stop_delta: params.early_stop_delta(),
                                            patience: params.patience(), equal_init:params.equal_init(), recompile:params.recompile(), e_weighted:params.e_weighted(), 
                                            learning_m:train::get_learningm_from_pylearningm(learning_m),})})
            }
        };
        Self{
            learner,
            params: get_param_from_pyparam(params),
        }
    }

    pub fn set_distributions(&mut self, distributions: Vec<f64>) {
        let mut paired_dist = vec![];
        for dist in distributions {
            paired_dist.push(vec![dist.ln(), (1.0-dist).ln()]);
        }
        self.learner.set_distributions(paired_dist, false);
    }

    pub fn get_distributions(&self) -> Vec<Vec<f64>> {
        self.learner.get_softmaxed_array().iter().map(|row| row.iter().map(|x| x.to_f64()).collect()).collect()
    }

    pub fn get_expected(&self) -> Vec<f64> {
        self.learner.get_train().get_expecteds().to_vec()
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

    pub fn loss_and_grad(&mut self, predictions: Vec<f64>, expected: Vec<f64>, mut idxs:Option<Vec<usize>>) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut train_loss = vec![0.0; expected.len()];
        let mut train_grad = vec![0.0; expected.len()];

        if idxs.is_none(){
            idxs = Some((0..expected.len()).collect());
        }
        let idxs = idxs.unwrap();
        for (i, id) in idxs.iter().enumerate() {
            train_loss[*id] = self.params.loss().loss(predictions[i], expected[*id]);
            train_grad[*id] = self.params.loss().gradient(predictions[i], expected[*id]);
        }
        self.learner.compute_gradients(&train_grad);
        (train_loss, self.learner.get_gradients().iter().map(|row| row.iter().map(|x| x.to_f64()).collect()).collect())
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