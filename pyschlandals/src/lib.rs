use pyo3::prelude::*;
use pyo3::Python;
use std::path::PathBuf;
use schlandals::*;

#[pyclass]
#[derive(Clone)]
enum BranchingHeuristic {
    MinInDegree,
    MinOutDegree,
    MaxDegree,
}

/// Submodule for exact search
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

/// Submodule for compilation
#[pymodule]
#[pyo3(name = "compiler")]
fn compilation_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "compiler")?;
    module.add_function(wrap_pyfunction!(compile_function, module)?)?;
    module.add_class::<PyDac>()?;
    module.add_class::<PyCircuitNode>()?;
    module.add_class::<PyDistributionNode>()?;

    parent_module.add_submodule(module)?;
    py.import("sys")?.getattr("modules")?.set_item("pyschlandals.compiler", module)?;
    Ok(())
}

#[pyclass(name = "Dac")]
struct PyDac {
    nodes: Vec<PyCircuitNode>,
    distributions: Vec<PyDistributionNode>,
    #[pyo3(get)]
    outputs: Vec<usize>,
    #[pyo3(get)]
    inputs: Vec<usize>,
}

#[pyclass(name = "CircuitNode")]
struct PyCircuitNode {
    #[pyo3(get)]
    outputs_start: usize,
    #[pyo3(get)]
    number_output: usize,
    #[pyo3(get)]
    inputs_start: usize,
    #[pyo3(get)]
    number_input: usize,
    #[pyo3(get)]
    distribution_input: Vec<(usize, usize)>,
    #[pyo3(get)]
    value: f64,
    #[pyo3(get)]
    is_mul: bool,
}

#[pyclass(name = "DistributionNode")]
struct PyDistributionNode {
    #[pyo3(get)]
    probabilities: Vec<f64>,
    #[pyo3(get)]
    outputs: Vec<(usize, usize)>,
}

#[pymethods]
impl PyDac {
    
    #[staticmethod]
    pub fn new() -> Self {
        let nodes: Vec<PyCircuitNode> = vec![];
        let distributions: Vec<PyDistributionNode> = vec![];
        let outputs: Vec<usize> = vec![];
        let inputs: Vec<usize> = vec![];
        Self {
            nodes,
            distributions,
            outputs,
            inputs,
        }
    }
    
    pub fn evaluate(&mut self) -> f64 {
        for i in 0..self.nodes.len() {
            self.nodes[i].value = if self.nodes[i].is_mul { 1.0 } else { 0.0 };
        }
        
        for i in 0..self.distributions.len() {
            for (output, value) in self.distributions[i].outputs.iter().copied() {
                let probability = self.distributions[i].probabilities[value];
                if self.nodes[output].is_mul {
                    self.nodes[output].value *= probability;
                } else {
                    self.nodes[output].value += probability;
                }
            }
        }
        
        for i in 0..self.nodes.len() {
            let start = self.nodes[i].outputs_start;
            let end = start + self.nodes[i].number_output;
            for j in start..end {
                let output = self.outputs[j];
                if self.nodes[output].is_mul {
                    self.nodes[output].value *= self.nodes[i].value;
                } else {
                    self.nodes[output].value += self.nodes[i].value;
                }
            }
        }
        self.nodes.last().unwrap().value
    }
    
    pub fn get_node_value(&self, node: usize) -> f64 {
        self.nodes[node].value
    }
    
    pub fn get_node_output_start(&self, node: usize) -> usize {
        self.nodes[node].outputs_start
    }
    
    pub fn get_node_input_start(&self, node: usize) -> usize {
        self.nodes[node].inputs_start
    }
    
    pub fn get_node_number_output(&self, node: usize) -> usize {
        self.nodes[node].number_output
    }
    
    pub fn get_node_number_input(&self, node: usize) -> usize {
        self.nodes[node].number_input
    }
    
    pub fn get_input_at(&self, index: usize) -> usize {
        self.inputs[index]
    }
    
    pub fn get_output_at(&self, index: usize) -> usize {
        self.outputs[index]
    }
    
    pub fn number_circuit_node(&self) -> usize {
        self.nodes.len()
    }
    
    pub fn number_distribution_node(&self) -> usize {
        self.distributions.len()
    }
    
    pub fn circuit_node_number_input_distribution(&self, node: usize) -> usize {
        self.nodes[node].distribution_input.len()
    }
    
    pub fn circuit_node_input_distribution_at(&self, node: usize, index: usize) -> (usize, usize) {
        self.nodes[node].distribution_input[index]
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
        Some(dac) => {
            let mut py_dac = PyDac::new();
            py_dac.outputs = dac.outputs_node_iter().map(|n| n.0).collect();
            py_dac.inputs = dac.inputs_node_iter().map(|n| n.0).collect();
            for distribution in dac.distributions_iter() {
                py_dac.distributions.push(PyDistributionNode {
                    probabilities: dac.get_distribution_probabilities(distribution).to_vec(),
                    outputs: dac.get_distribution_outputs(distribution).iter().copied().map(|(node, value)| (node.0, value)).collect(),
                })
            }
            
            for node in dac.nodes_iter() {
                let is_mul = dac.is_circuit_node_mul(node);
                py_dac.nodes.push(PyCircuitNode {
                    outputs_start: dac.get_circuit_node_out_start(node),
                    number_output: dac.get_circuit_node_number_output(node),
                    inputs_start: dac.get_circuit_node_in_start(node),
                    number_input: dac.get_circuit_node_number_input(node),
                    distribution_input: dac.get_circuit_node_input_distribution(node).map(|(n, v)| (n.0, v)).collect(),
                    value: if is_mul { 1.0 } else { 0.0 },
                    is_mul,
                })
            }
            Some(py_dac)
        }
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