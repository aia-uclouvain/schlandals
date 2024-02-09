use pyo3::prelude::*;
use pyo3::Python;
use std::path::PathBuf;
use schlandals::*;

mod train;
use train::*;

#[pyclass]
#[derive(Clone, Copy)]
pub enum PyBranching {
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
struct PyProblem {
    distributions: Vec<Vec<f64>>,
    clauses: Vec<Vec<isize>>,
    branching: PyBranching,
    epsilon: f64,
    timeout: u64,
    memory_limit: u64,
    statistics: bool,
}

#[pymethods]
impl PyProblem {

    #[new]
    pub fn new(branching: Option<PyBranching>, epsilon: Option<f64>, timeout: Option<u64>, memory_limit: Option<u64>, statistics: Option<bool>) -> Self {
        Self {
            distributions: vec![],
            clauses: vec![],
            branching: if let Some(b) = branching { b } else { PyBranching::MinInDegree },
            epsilon: if let Some(e) = epsilon { e } else { 0.0 },
            timeout: if let Some(to) = timeout { to } else { u64::MAX },
            memory_limit: if let Some(limit) = memory_limit { limit } else { u64::MAX },
            statistics: if let Some(s) = statistics { s } else { false },
        }
    }

    pub fn add_distribution(&mut self, distribution: Vec<f64>) {
        self.distributions.push(distribution);
    }

    pub fn add_clause(&mut self, clause: Vec<isize>) {
        self.clauses.push(clause);
    }

    pub fn solve(&self) -> Option<f64> {
        match schlandals::solve_from_problem(&self.distributions, &self.clauses, get_branching_from_pybranching(self.branching), self.epsilon, Some(self.memory_limit), self.timeout, self.statistics) {
            Ok(p) => Some(p.to_f64()),
            Err(e) => {
                println!("{:?}", e);
                None
            }
        }
    }

    pub fn compile(&self, fdac: Option<String>, dotfile: Option<String>) -> Option<f64> {
        match schlandals::compile_from_problem(&self.distributions,
                                               &self.clauses,
                                               get_branching_from_pybranching(self.branching),
                                               self.epsilon,
                                               Some(self.memory_limit),
                                               self.timeout,
                                               self.statistics,
                                               if let Some(path) = fdac { Some(PathBuf::from(path)) } else { None },
                                               if let Some(path) = dotfile { Some(PathBuf::from(path)) } else { None },) {
            Ok(p) => Some(p.to_f64()),
            Err(e) => {
                println!("{:?}", e);
                None
            }
        }
    }
}


#[pyfunction]
#[pyo3(name = "search")]
fn pysearch(file: String, branching: PyBranching, epsilon: Option<f64>, memory_limit: Option<u64>, timeout: Option<u64>) -> Option<f64> {
    let e = if epsilon.is_none() {
        0.0
    } else {
        epsilon.unwrap()
    };
    let to = if timeout.is_none() { u64::MAX } else { timeout.unwrap() };
    match schlandals::search(PathBuf::from(file), get_branching_from_pybranching(branching), false, memory_limit, e, to) {
        Err(_) => None,
        Ok(p) => Some(p.to_f64()),
    }
}

#[pyfunction]
#[pyo3(name = "compile")]
fn pycompile(file: String, branching: PyBranching, epsilon: Option<f64>, output_circuit: Option<String>, output_dot: Option<String>, timeout: Option<u64>) -> Option<f64> {
    let fdac = if let Some(file) = output_circuit { Some(PathBuf::from(file)) } else { None };
    let fdot = if let Some(file) = output_dot { Some(PathBuf::from(file)) } else { None };
    let e = if let Some(e) = epsilon { e } else { 0.0 };
    let to = if timeout.is_none() { u64::MAX } else { timeout.unwrap() };
    match schlandals::compile(PathBuf::from(file), get_branching_from_pybranching(branching), fdac, fdot, e, to) {
        Err(_) => None,
        Ok(p) => Some(p.to_f64()),
    }
}

#[pymodule]
#[pyo3(name="pwmc")]
fn pwmc_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "pwmc")?;
    module.add_class::<PyProblem>()?;
    module.add_function(wrap_pyfunction!(pycompile, module)?)?;
    module.add_function(wrap_pyfunction!(pysearch, module)?)?;

    parent_module.add_submodule(module)?;
    py.import("sys")?.getattr("modules")?.set_item("pyschlandals.pwmc", module)?;
    Ok(())
}

/// Base module for pyschlandals
#[pymodule]
fn pyschlandals(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBranching>()?;
    pwmc_submodule(py, m)?;
    train::learn_submodule(py, m)?;
    Ok(())
}
