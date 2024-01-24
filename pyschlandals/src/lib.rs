use pyo3::prelude::*;
use pyo3::Python;
use std::path::PathBuf;
use schlandals::*;

#[pyclass]
#[derive(Clone)]
enum PyBranching {
    MinInDegree,
    MinOutDegree,
    MaxDegree,
}

fn get_branching_from_pybranching(branching: PyBranching) -> Branching {
    match branching {
        PyBranching::MinInDegree => Branching::MinInDegree,
        PyBranching::MinOutDegree => Branching::MinOutDegree,
        PyBranching::MaxDegree => Branching::MaxDegree,
    }
}

#[pyclass]
#[derive(Clone)]
enum PyLoss {
    MAE,
    MSE,
}

fn get_loss_from_pyloss(loss: PyLoss) -> Loss {
    match loss {
        PyLoss::MAE => Loss::MAE,
        PyLoss::MSE => Loss::MSE,
    }
}

// TODO: Find how to make the python binding to take into account that the tensors are a feature
// that is not enabled by default
#[pyclass]
#[derive(Clone)]
enum PySemiring {
    Probability,
    //Tensor,
}

fn get_semiring_from_pysemiring(semiring: PySemiring) -> Semiring {
    match semiring {
        PySemiring::Probability => Semiring::Probability,
        //PySemiring::Tensor => Semiring::Tensor,
    }
}

#[pyclass]
#[derive(Clone)]
enum PyOptimizer {
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
#[pyo3(name = "search")]
fn pysearch(file: String, branching: PyBranching) -> Option<f64> {
    match schlandals::search(PathBuf::from(file), get_branching_from_pybranching(branching), false, None, 0.0) {
        Err(_) => None,
        Ok(p) => Some(p.to_f64()),
    }
}

#[pyfunction]
#[pyo3(name = "learn")]
fn pylearn(train_file: String, branching: PyBranching, learning_rate: f64, nepochs: usize, log: bool, timeout: u64, epsilon: f64, loss: PyLoss, jobs: usize, semiring: PySemiring, optimizer: PyOptimizer, test_file: Option<String>, outfolder: Option<PathBuf>) {
    let b = get_branching_from_pybranching(branching);
    let l = get_loss_from_pyloss(loss);
    let s = get_semiring_from_pysemiring(semiring);
    let o = get_optimizer_from_pyoptimizer(optimizer);
    let train = PathBuf::from(train_file);
    let test = if test_file.is_none() {
        None
    } else {
        Some(PathBuf::from(test_file.unwrap()))
    };
    schlandals::learn(train, test, b, outfolder, learning_rate, nepochs, log, timeout, epsilon, l, jobs, s, o);
}


/// Base module for pyschlandals
#[pymodule]
fn pyschlandals(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBranching>()?;
    m.add_class::<PyLoss>()?;
    m.add_class::<PyOptimizer>()?;
    m.add_class::<PySemiring>()?;
    m.add_function(wrap_pyfunction!(pylearn, m)?).unwrap();
    m.add_function(wrap_pyfunction!(pysearch, m)?).unwrap();
    Ok(())
}
