use pyo3::prelude::*;
use pyo3::Python;
use std::path::PathBuf;
use schlandals::common::Branching;
mod train;
use train::*;
mod learner;
use learner::*;

#[pyclass]
#[derive(Clone, Copy)]
pub enum PyBranching {
    MinInDegree,
}

fn get_branching_from_pybranching(branching: PyBranching) -> Branching {
    match branching {
        PyBranching::MinInDegree => Branching::MinInDegree,
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

    pub fn solve(&self) -> (f64, f64) {
        let mut schlandals_arg = schlandals::Args::default();
        schlandals_arg.timeout = self.timeout;
        schlandals_arg.branching = get_branching_from_pybranching(self.branching);
        schlandals_arg.statistics = self.statistics;
        schlandals_arg.memory = self.memory_limit;
        schlandals_arg.epsilon = self.epsilon;
        schlandals::pysearch(schlandals_arg, &self.distributions, &self.clauses)
    }

    pub fn to_cnf(&self) -> String {
        let mut cnf = String::new();
        let number_variables = *self.clauses.iter().map(|c| c.iter().max().unwrap()).max().unwrap();
        let number_clauses = self.clauses.len();
        cnf.push_str(&format!("p cnf {} {}\n", number_variables, number_clauses));
        for distribution in self.distributions.iter() {
            let dstr = distribution.iter().map(|f| format!("{}", f)).collect::<Vec<String>>().join(" "); 
            cnf.push_str(&format!("c p distribution {}\n", dstr))
        }
        for clause in self.clauses.iter() {
            let cstr = clause.iter().map(|l| format!("{}", l)).collect::<Vec<String>>().join(" ");
            cnf.push_str(&format!("{} 0\n", cstr));
        }
        cnf
    }

    pub fn copy(&self) -> PyProblem {
        self.clone()
    }
}

#[pymodule]
#[pyo3(name="pwmc")]
fn pwmc_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "pwmc")?;
    module.add_class::<PyProblem>()?;
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
