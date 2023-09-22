use pyo3::prelude::*;
use pyo3::Python;
use std::path::PathBuf;
use schlandals::*;

#[pyclass]
#[derive(Clone)]
/// Available branching heuristic for the solver
enum BranchingHeuristic {
    /// Selects a distribution from a clause with the minimum in-degree in the implication graph
    MinInDegree,
    /// Selects a distribution from a clause with the minimum out-degree in the implication graph
    MinOutDegree,
    /// Selects a distribution from a clause with the maximum degree in the implication graph
    MaxDegree,
}

#[pymodule]
#[pyo3(name = "search")]
fn exact_search_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "search")?;
    module.add_function(wrap_pyfunction!(search_function, module)?)?;
    module.add_function(wrap_pyfunction!(approximate_search_function, module)?)?;

    parent_module.add_submodule(module)?;
    py.import("sys")?.getattr("modules")?.set_item("pyschlandals.search", module)?;
    Ok(())
}

#[pyfunction]
#[pyo3(name = "exact")]
fn search_function(file: String, branching: BranchingHeuristic) -> Option<f64> {
    let branching_heuristic: Branching = match branching {
        BranchingHeuristic::MinInDegree => Branching::MinInDegree,
        BranchingHeuristic::MinOutDegree => Branching::MinOutDegree,
        BranchingHeuristic::MaxDegree => Branching::MaxDegree,
    };
    match schlandals::search(PathBuf::from(file), branching_heuristic, false, None) {
        None => None,
        Some(p) => Some(p.to_f64()),
    }
}

#[pyfunction]
#[pyo3(name = "approximate")]
fn approximate_search_function(file: String, branching: BranchingHeuristic, epsilon: f64) -> Option<f64> {
    let branching_heuristic: Branching = match branching {
        BranchingHeuristic::MinInDegree => Branching::MinInDegree,
        BranchingHeuristic::MinOutDegree => Branching::MinOutDegree,
        BranchingHeuristic::MaxDegree => Branching::MaxDegree,
    };
    match schlandals::approximate_search(PathBuf::from(file), branching_heuristic, false, None, epsilon) {
        None => None,
        Some(p) => Some(p.to_f64()),
    }
}

#[pymodule]
#[pyo3(name = "compiler")]
fn compilation_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "compiler")?;
    module.add_function(wrap_pyfunction!(compile_function, module)?)?;
    module.add_class::<PyDac>()?;

    parent_module.add_submodule(module)?;
    py.import("sys")?.getattr("modules")?.set_item("pyschlandals.compiler", module)?;
    Ok(())
}

#[pyclass(name = "Dac")]
/// Python-exposed structure representing the distributed aware arithmetic circuit
struct PyDac {
    dac: Dac,
}

#[pymethods]
impl PyDac {
    
    #[staticmethod]
    /// Parse the given file and create the dac from it.
    pub fn from_fdac_file(file: String) -> Self {
        let dac = Dac::from_file(&PathBuf::from(file));
        PyDac { dac }
    }
    
    /// Evaluates the circuit and returns the computed probability
    pub fn evaluate(&mut self) -> f64 {
        self.dac.evaluate().to_f64()
    }
    
    /// Returns the probability computed by the circuit
    pub fn get_circuit_probability(&self) -> f64 {
        self.dac.get_circuit_probability().to_f64()
    }
    
    /// Returns the probability computed at a node
    pub fn get_node_value(&self, node: usize) -> f64 {
        self.dac.get_circuit_node_probability(CircuitNodeIndex(node)).to_f64()
    }
    
    /// Returns true if and only if the node is a multiplicative node
    pub fn is_node_mul(&self, node: usize) -> bool {
        self.dac.is_circuit_node_mul(CircuitNodeIndex(node))
    }
    
    /// Returns the first index, in the output vector, of the outpouts of
    /// the given node
    pub fn get_node_output_start(&self, node: usize) -> usize {
        self.dac.get_circuit_node_out_start(CircuitNodeIndex(node))
    }
    
    /// Returns the first index, in the input vector, of the inputs of
    /// the given node
    pub fn get_node_input_start(&self, node: usize) -> usize {
        self.dac.get_circuit_node_in_start(CircuitNodeIndex(node))
    }
    
    /// Returns the number of outputs of the given node
    pub fn get_node_number_output(&self, node: usize) -> usize {
        self.dac.get_circuit_node_number_output(CircuitNodeIndex(node))
    }
    
    /// Returns the number of inputs of the given node
    pub fn get_node_number_input(&self, node: usize) -> usize {
        self.dac.get_circuit_node_number_input(CircuitNodeIndex(node))
    }
    
    /// Returns the node's index from the input vector at the given index
    pub fn get_input_at(&self, index: usize) -> usize {
        self.dac.get_input_at(index).0
    }
    
    /// Returns the node's index from the output vector at the given index
    pub fn get_output_at(&self, index: usize) -> usize {
        self.dac.get_output_at(index).0
    }
    
    /// Returns the number of comoutational nodes in the circuit
    pub fn number_circuit_node(&self) -> usize {
        self.dac.number_nodes()
    }
    
    /// Returns the number of distribution nodes in the circuit
    pub fn number_distribution_node(&self) -> usize {
        self.dac.number_distributions()
    }
    
    /// Returns the number of distribution-inputs of the given node
    pub fn circuit_node_number_input_distribution(&self, node: usize) -> usize {
        self.dac.get_circuit_node_number_distribution_input(CircuitNodeIndex(node))
    }
    
    /// Returns a particular (distribution, value)-pair from the node distribution-inputs
    pub fn circuit_node_input_distribution_at(&self, node: usize, index: usize) -> (usize, usize) {
        let x = self.dac.get_circuit_node_input_distribution_at(CircuitNodeIndex(node), index);
        (x.0.0, x.1)
    }
    
    /// Returns the size of the domain of the distribution
    pub fn get_distribution_number_value(&self, distribution: usize) -> usize {
        self.dac.get_distribution_domain_size(DistributionNodeIndex(distribution))
    }
    
    /// Returns the pair (circuit node, value index) of the output of the distribution at its given output-index
    pub fn get_distribution_node_output_at(&self, distribution: usize, index: usize) -> (usize, usize) {
        let x = self.dac.get_distribution_output_at(DistributionNodeIndex(distribution), index);
        (x.0.0, x.1)
    }
    
    /// Returns the number of output of a distribution node
    pub fn get_distribution_number_output(&self, distribution: usize) -> usize {
        self.dac.get_distribution_number_output(DistributionNodeIndex(distribution))
    }
    
    /// Returns the probability, of the given distribution, at the given index 
    pub fn get_distribution_probability(&self, distribution: usize, probability_index: usize) -> f64 {
    self.dac.get_distribution_probability_at(DistributionNodeIndex(distribution), probability_index)
    }
    
    /// Sets the probability at the given index in the given distribution
    pub fn set_distribution_probability(&mut self, distribution: usize, probability_index: usize, probability: f64) {
        self.dac.set_distribution_probability_at(DistributionNodeIndex(distribution), probability_index, probability);
    }
    
    /// Returns the graphviz representation of the circuit
    pub fn to_graphviz(&self) -> String {
        self.dac.as_graphviz()
    }
}

#[pyfunction]
#[pyo3(name = "compile")]
fn compile_function(file: String, branching: BranchingHeuristic) -> Option<PyDac> {
    let branching_heuristic: Branching = match branching {
        BranchingHeuristic::MinInDegree => Branching::MinInDegree,
        BranchingHeuristic::MinOutDegree => Branching::MinOutDegree,
        BranchingHeuristic::MaxDegree => Branching::MaxDegree,
    };
    match compile(PathBuf::from(file), branching_heuristic, None, None) {
        None => None,
        Some(dac) => Some(PyDac { dac }),
    }
}

/// Base module for pyschlandals
#[pymodule]
fn pyschlandals(py: Python, m: &PyModule) -> PyResult<()> {
    exact_search_submodule(py, m)?;
    compilation_submodule(py, m)?;
    m.add_class::<BranchingHeuristic>()?;
    Ok(())
}