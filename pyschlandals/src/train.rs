use pyo3::prelude::*;
use super::*;

use schlandals::learning::LearnParameters;
use schlandals::ApproximateMethod;
use schlandals::Loss;
use schlandals::Semiring;
use schlandals::Optimizer;
use schlandals::Command;

#[pyclass]
#[derive(Clone)]
pub enum PyLoss {
    MAE,
    MSE,
}

fn get_loss_from_pyloss(loss: PyLoss) -> Loss {
    match loss {
        PyLoss::MAE => Loss::MAE,
        PyLoss::MSE => Loss::MSE,
    }
}


#[pyclass]
#[derive(Clone)]
pub struct PyLearnParameters {
    /// The initial learning rate
    lr: f64,
    /// The number of epochs
    nepochs: usize,
    /// The timeout used for the compilations, in seconds
    compilation_timeout: u64,
    /// The timeout used for the training loop, in seconds
    learn_timeout: u64,
    /// The loss function
    loss: PyLoss,
    /// The optimizer
    optimizer: PyOptimizer,
    /// The learning rate decay
    lr_drop: f64,
    /// The number of epochs after which the learning rate is dropped
    epoch_drop: usize,
    /// The error threshold under which the training is stopped
    early_stop_threshold: f64,
    /// The minimum delta between two epochs to consider that the training is still improving
    early_stop_delta: f64,
    /// The number of epochs to wait before stopping the training if the loss is not improving
    patience: usize,
}

#[pymethods]
impl  PyLearnParameters {

    #[new]
    pub fn new(lr: Option<f64>, nepochs: Option<usize>, compilation_timeout: Option<u64>, learn_timeout: Option<u64>, loss: Option<PyLoss>, optimizer: Option<PyOptimizer>, lr_drop: Option<f64>, epoch_drop: Option<usize>, early_stop_threshold: Option<f64>, early_stop_delta: Option<f64>, patience: Option<usize>) -> Self {
        Self {
            lr: if let Some(v) = lr { v } else { 0.3 },
            nepochs: if let Some(v) = nepochs { v } else { 6000 },
            compilation_timeout: if let Some(v) = compilation_timeout { v } else { u64::MAX },
            learn_timeout: if let Some(v) = learn_timeout { v }  else { u64::MAX },
            loss: if let Some(v) = loss { v } else { PyLoss::MAE },
            optimizer: if let Some(v) = optimizer { v } else { PyOptimizer::Adam },
            lr_drop: if let Some(v) = lr_drop { v } else { 0.75 },
            epoch_drop: if let Some(v) = epoch_drop { v } else { 100 },
            early_stop_threshold: if let Some(v) = early_stop_threshold { v } else { 0.0001 },
            early_stop_delta: if let Some(v) = early_stop_delta { v } else { 0.00001 },
            patience: if let Some(v) = patience { v } else { 5 },
        }
    }
}

fn get_param_from_pyparam(param: PyLearnParameters) -> LearnParameters {
    LearnParameters::new(param.lr,
                         param.nepochs,
                         param.compilation_timeout,
                         param.learn_timeout,
                         get_loss_from_pyloss(param.loss),
                         get_optimizer_from_pyoptimizer(param.optimizer),
                         param.lr_drop,
                         param.epoch_drop,
                         param.early_stop_threshold,
                         param.early_stop_delta,
                         param.patience,
                         false,
                         false)
}

// TODO: Find how to make the python binding to take into account that the tensors are a feature
// that is not enabled by default
#[pyclass]
#[derive(Clone)]
pub enum PySemiring {
    Probability,
}

fn get_semiring_from_pysemiring(semiring: PySemiring) -> Semiring {
    match semiring {
        PySemiring::Probability => Semiring::Probability,
    }
}

#[pyclass]
#[derive(Clone)]
pub enum PyOptimizer {
    SGD,
    Adam,
}

fn get_optimizer_from_pyoptimizer(optimizer: PyOptimizer) -> Optimizer {
    match optimizer {
        PyOptimizer::SGD => Optimizer::SGD,
        PyOptimizer::Adam => Optimizer::Adam,
    }
}

#[pyfunction]
#[pyo3(name = "learn")]
pub fn pylearn(model: String, train_file: String, param: PyLearnParameters, branching: Option<PyBranching>, semiring: Option<PySemiring>, log: Option<bool>, epsilon: Option<f64>, jobs: Option<usize>, test_file: Option<String>, outfolder: Option<PathBuf>) {
    let mut schlandals_arg = schlandals::Args::default();
    schlandals_arg.branching = get_branching_from_pybranching(branching.unwrap_or(PyBranching::MinInDegree));
    schlandals_arg.epsilon = epsilon.unwrap_or(0.0);
    let mut cmd = Command::default_learn_args();
    let train = PathBuf::from(train_file);
    schlandals::learn(schlandals_arg);
}

#[pymodule]
#[pyo3(name="learn")]
pub fn learn_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "learn")?;

    module.add_class::<PyLearnParameters>()?;
    module.add_class::<PyLoss>()?;
    module.add_class::<PyOptimizer>()?;
    module.add_class::<PySemiring>()?;
    module.add_function(wrap_pyfunction!(pylearn, module)?)?;

    parent_module.add_submodule(module)?;
    py.import("sys")?.getattr("modules")?.set_item("pyschlandals.learn", module)?;
    Ok(())
}

