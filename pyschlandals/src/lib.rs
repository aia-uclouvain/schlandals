use pyo3::prelude::*;
use pyo3::Python;
use schlandals::*;
use schlandals::core::graph::DistributionIndex;
use schlandals::diagrams::dac::dac::{NodeIndex, Dac};
use schlandals::learning::LogLearner;
use schlandals::learning::learner::DacIndex;
use std::{path::PathBuf, fs::File, io::{BufRead,BufReader}};
use rug::Float;

#[pyclass]
#[derive(Clone)]
/// Available branching heuristic for the solver
enum BranchingHeuristic {
    /// Selects a distribution from a clause with the minimum in-degree in the implication graph
    MinInDegree,
    /// Selects a distribution from a clause with the minimum out-degree in the implication graph
    MinOutDegree,
    /// Selects a distBranchingHeuristicribution from a clause with the maximum degree in the implication graph
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
    match schlandals::search(PathBuf::from(file), branching_heuristic, false, None, 0.0) {
        Err(_) => None,
        Ok(p) => Some(p.to_f64()),
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
    match schlandals::search(PathBuf::from(file), branching_heuristic, false, None, epsilon) {
        Err(_) => None,
        Ok(p) => Some(p.to_f64()),
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

#[pymodule]
#[pyo3(name = "learning")]
fn learning_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "learning")?;
    module.add_class::<PyLearner>()?;

    parent_module.add_submodule(module)?;
    py.import("sys")?.getattr("modules")?.set_item("pyschlandals.learning", module)?;
    Ok(())
}

#[pyclass(name = "Dac")]
/// Python-exposed structure representing the distributed aware arithmetic circuit
struct PyDac {
    dac: Dac,
}

#[pyclass(name = "Learner")]
struct PyLearner {
    learner: LogLearner,
}

#[pymethods]
impl PyLearner {
    #[staticmethod]
    /// Parse the given folder and create a learner from it.
    pub fn create_learner(trainfile: String, branching: BranchingHeuristic, outputdir: String) -> Self {
        let path_trainfile = PathBuf::from(trainfile);
        let mut inputs = vec![];
        let mut expected: Vec<f64> = vec![];
        let file = File::open(&path_trainfile).unwrap();
        let reader = BufReader::new(file);
        for line in reader.lines().skip(1) {
            let l = line.unwrap();
            let mut split = l.split(",");
            inputs.push(path_trainfile.parent().unwrap().join(split.next().unwrap().parse::<PathBuf>().unwrap()));
            expected.push(split.next().unwrap().parse::<f64>().unwrap());
        }
        let branching_heuristic: Branching = match branching {
            BranchingHeuristic::MinInDegree => Branching::MinInDegree,
            BranchingHeuristic::MinOutDegree => Branching::MinOutDegree,
            BranchingHeuristic::MaxDegree => Branching::MaxDegree,
        };
        let outfolder = PathBuf::from(outputdir);
        let learner = LogLearner::new(inputs, expected, 0.0, branching_heuristic, Some(outfolder), 1.0);
        PyLearner { learner }
    }

    /// Evaluates the different dacs and returns the computed probabilities
    pub fn evaluate(&mut self) -> Vec<f64> {
        self.learner.evaluate()
    }

    pub fn start_logger(&mut self) {
        self.learner.start_logger();
    }

    pub fn log_epoch(&mut self, pred_distribs: Vec<Vec<f64>>, loss:Vec<f64>, gradients: Vec<Vec<f64>>, lr:f64) {
        let grads: Vec<Vec<Float>> = gradients.iter().map(|x| x.iter().map(|y| Float::with_val(128, *y)).collect()).collect();
        self.learner.log_epoch(&pred_distribs, &loss, &grads, lr);
    }

    /// Returns the expected outputs
    pub fn get_expected_outputs(&self) -> Vec<f64> {
        self.learner.get_expected_outputs().clone()
    }

    /// Returns the real weights
    pub fn get_expected_distributions(&self) -> Vec<Vec<f64>> {
        self.learner.get_expected_distributions().clone()
    }

    /// Returns the current unsoftmaxed weights
    pub fn get_current_distributions(&self) -> Vec<Vec<f64>> {
        self.learner.get_current_distributions().clone()
    }

    /// Returns the current softmaxed weights
    pub fn get_current_softmaxed_distributions(&self) -> Vec<Vec<f64>> {
        self.learner.get_softmaxed_array().clone()
    }

    /// Returns the number of dacs
    pub fn get_number_dacs(&self) -> usize {
        self.learner.get_number_dacs()
    }

    /// Returns the number of circuit nodes for dac i
    pub fn get_number_circuit_nodes(&self, i: usize) -> usize {
        self.learner[DacIndex(i)].get_number_circuit_nodes()
    }

    /// Returns the number of distributions nodes for dac i
    pub fn get_number_distribution_nodes(&self, i: usize) -> usize {
        self.learner[DacIndex(i)].get_number_distribution_nodes()
    }

    /// Returns the number of inputs of the given node for dac i
    pub fn get_node_number_input(&self, i: usize, node: usize) -> usize {
        self.learner[DacIndex(i)][NodeIndex(node)].get_number_inputs()
    }

    /// Returns the node's index from the input vector at the given index for dac i
    pub fn get_input_at(&self, i: usize, index: usize) -> usize {
        self.learner[DacIndex(i)].get_input_at(index).0
    }

    /// Returns the first index, in the input vector, of the inputs of
    /// the given node for dac i
    pub fn get_node_input_start(&self, i: usize, node: usize) -> usize {
        self.learner[DacIndex(i)][NodeIndex(node)].get_input_start()
    }

    /// Returns the CiruitIndex of the distribution distribution and value for dac i
    pub fn get_distribution_value_node_index(&self, i: usize, distribution: usize, value:usize) -> isize {
        self.learner[DacIndex(i)].get_distribution_value_node_index_usize(DistributionIndex(distribution), value)
    }

    /// Returns the number of outputs of the given node for dac i
    pub fn get_node_number_output(&self, i: usize, node: usize) -> usize {
        self.learner[DacIndex(i)][NodeIndex(node)].get_number_outputs()
    }

    /// Returns the node's index from the output vector at the given index for dac i
    pub fn get_output_at(&self, i: usize, index: usize) -> usize {
        self.learner[DacIndex(i)].get_output_at(index).0
    }

    /// Returns the start index of the outputs of the given node for dac i
    pub fn get_node_output_start(&self, i: usize, node: usize) -> usize {
        self.learner[DacIndex(i)][NodeIndex(node)].get_output_start()
    }

    /// Returns if the node is a distribution node for dac i
    pub fn is_node_distribution(&self, i: usize, node: usize) -> bool {
        self.learner[DacIndex(i)][NodeIndex(node)].is_distribution()
    }

    /// Returns if the node is a multiplication node for dac i
    pub fn is_node_mul(&self, i: usize, node: usize) -> bool {
        self.learner[DacIndex(i)][NodeIndex(node)].is_product()
    }

    /// prints the graphviz representation of the dac i
    pub fn to_graphviz(&self, i: usize) {
        println!("{}",self.learner[DacIndex(i)].as_graphviz())
    }
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

    /// Computes the gradient of each distribution, layer by layer (starting from the root, to the distributions)
    /* pub fn compute_grads(&mut self, grad_loss: f64, lr: f64){
        self.dac.compute_grads(grad_loss, lr);
    } */
    
    /// Returns the probability computed by the circuit
    pub fn get_circuit_probability(&self) -> f64 {
        self.dac.get_circuit_probability().to_f64()
    }
    
    /// Returns the probability computed at a node
    pub fn get_node_value(&self, node: usize) -> f64 {
        self.dac[NodeIndex(node)].get_value().to_f64()
    }
    
    /// Returns true if and only if the node is a multiplicative node
    pub fn is_node_mul(&self, node: usize) -> bool {
        self.dac[NodeIndex(node)].is_product()
    }
    
    /// Returns the first index, in the output vector, of the outpouts of
    /// the given node
    pub fn get_node_output_start(&self, node: usize) -> usize {
        self.dac[NodeIndex(node)].get_output_start()
    }
    
    /// Returns the first index, in the input vector, of the inputs of
    /// the given node
    pub fn get_node_input_start(&self, node: usize) -> usize {
        self.dac[NodeIndex(node)].get_input_start()
    }
    
    /// Returns the number of outputs of the given node
    pub fn get_node_number_output(&self, node: usize) -> usize {
        self.dac[NodeIndex(node)].get_number_outputs()
    }
    
    /// Returns the number of inputs of the given node
    pub fn get_node_number_input(&self, node: usize) -> usize {
        self.dac[NodeIndex(node)].get_number_inputs()
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
        self.dac.get_circuit_node_number_distribution_input(NodeIndex(node))
    }
    
    /// Returns a particular (distribution, value)-pair from the node distribution-inputs
    /* pub fn circuit_node_input_distribution_at(&self, node: usize, index: usize) -> (usize, usize) {
        let x = self.dac.get_circuit_node_input_distribution_at(NodeIndex(node), index);
        (x.0.0, x.1)
    } */
    
    /// Returns the size of the domain of the distribution
    /* pub fn get_distribution_number_value(&self, distribution: usize) -> usize {
        self.dac.get_distribution_domain_size(NodeIndex(distribution))
    } */
    
    /// Returns the pair (circuit node, value index) of the output of the distribution at its given output-index
    /* pub fn get_distribution_node_output_at(&self, distribution: usize, index: usize) -> (usize, usize) {
        let x = self.dac.get_distribution_output_at(NodeIndex(distribution), index);
        (x.0.0, x.1)
    } */
    
    /// Returns the number of output of a distribution node
    /* pub fn get_distribution_number_output(&self, distribution: usize) -> usize {
        self.dac.get_distribution_number_output(NodeIndex(distribution))
    } */
    
    /// Returns the probability, of the given distribution, at the given index 
    /* pub fn get_distribution_probability(&self, distribution: usize, probability_index: usize) -> f64 {
    self.dac.get_distribution_probability_at(NodeIndex(distribution), probability_index)
    } */
    
    /* pub fn get_distribution_gradient_at(&self, distribution: usize, probability_index: usize) -> f64 {
        self.dac.get_distribution_gradient_at(NodeIndex(distribution), probability_index)
    } */

    /// Sets the unsoftmaxed probability at the given index in the given distribution
    /* pub fn set_distribution_probability(&mut self, distribution: usize, probability_index: usize, probability: f64) {
        self.dac.set_distribution_probability_at(NodeIndex(distribution), probability_index, probability);
    } */
    
    /// Returns the graphviz representation of the circuit
    pub fn to_graphviz(&self) -> String {
        self.dac.as_graphviz()
    }
}

#[pyfunction]
#[pyo3(name = "compile")]
fn compile_function(file: String, branching: BranchingHeuristic, ratio:Option<f64>) -> Option<PyDac> {
    let branching_heuristic: Branching = match branching {
        BranchingHeuristic::MinInDegree => Branching::MinInDegree,
        BranchingHeuristic::MinOutDegree => Branching::MinOutDegree,
        BranchingHeuristic::MaxDegree => Branching::MaxDegree,
    };
    if let Some(r) = ratio {
        match compile(PathBuf::from(file), branching_heuristic, r, None, None) {
            None => None,
            Some(dac) => Some(PyDac { dac }),
        }
    } else {
        match compile(PathBuf::from(file), branching_heuristic, 1.0, None, None) {
            None => None,
            Some(dac) => Some(PyDac { dac }),
        }
    }
}


/// Base module for pyschlandals
#[pymodule]
fn pyschlandals(py: Python, m: &PyModule) -> PyResult<()> {
    exact_search_submodule(py, m)?;
    compilation_submodule(py, m)?;
    learning_submodule(py, m)?;
    m.add_class::<BranchingHeuristic>()?;
    Ok(())
}