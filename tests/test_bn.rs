use schlandals;
use schlandals::branching::FirstBranching;
use schlandals::components::DFSComponentExtractor;
use schlandals::ppidimacs::graph_from_ppidimacs;
use schlandals::solver::solver::Solver;
use schlandals::trail::TrailedStateManager;

use std::path::PathBuf;

use assert_float_eq::*;

#[test]
fn test_abc_chain_a0() {
    assert_f64_near!(
        0.2_f64.log2(),
        solve_instance("tests/instances/bayesian_networks/abc_chain_a0.ppidimacs")
    );
}

#[test]
fn test_abc_chain_a1() {
    assert_f64_near!(
        0.8_f64.log2(),
        solve_instance("tests/instances/bayesian_networks/abc_chain_a1.ppidimacs")
    );
}

#[test]
fn test_abc_chain_b0() {
    assert_f64_near!(
        0.38_f64.log2(),
        solve_instance("tests/instances/bayesian_networks/abc_chain_b0.ppidimacs")
    );
}

#[test]
fn test_abc_chain_b1() {
    assert_f64_near!(
        0.62_f64.log2(),
        solve_instance("tests/instances/bayesian_networks/abc_chain_b1.ppidimacs")
    );
}

#[test]
fn test_abc_chain_c0() {
    assert_f64_near!(
        0.348_f64.log2(),
        solve_instance("tests/instances/bayesian_networks/abc_chain_c0.ppidimacs")
    );
}

#[test]
fn test_abc_chain_c1() {
    assert_f64_near!(
        0.652_f64.log2(),
        solve_instance("tests/instances/bayesian_networks/abc_chain_c1.ppidimacs")
    );
}
fn solve_instance(filename: &'static str) -> f64 {
    let mut state = TrailedStateManager::new();
    let path = PathBuf::from(filename);
    let (graph, v) = graph_from_ppidimacs(&path, &mut state).unwrap();
    let component_extractor = DFSComponentExtractor::new(&graph, &mut state);
    let branching_heuristic = FirstBranching::default();
    let mut solver: Solver<TrailedStateManager, DFSComponentExtractor, FirstBranching> =
        Solver::new(graph, state, component_extractor, branching_heuristic);
    solver.solve(v)
}
