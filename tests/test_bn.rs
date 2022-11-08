use schlandals;
use schlandals::branching::*;
use schlandals::components::*;
use schlandals::ppidimacs::graph_from_ppidimacs;
use schlandals::solver::sequential::Solver;
use schlandals::trail::StateManager;

use std::path::PathBuf;

use assert_float_eq::*;

macro_rules! integration_tests_bn {
    ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let filename = format!("tests/instances/bayesian_networks/{}.ppidimacs", stringify!($name));
                let mut state = StateManager::default();
                let path = PathBuf::from(filename);
                let (graph, v) = graph_from_ppidimacs(&path, &mut state).unwrap();
                let component_extractor = ComponentExtractor::new(&graph, &mut state);
                //let branching_heuristic = FirstBranching::default();
                //let mut branching_heuristic = ActiveDegreeBranching::default();
                let mut branching_heuristic = Articulation::default();
                let mut solver = Solver::new(graph, state, component_extractor, &mut branching_heuristic);
                let sol = solver.solve(v);
                assert_float_relative_eq!($value, 2_f64.powf(sol), 0.000001);
            }
        )*
    }
}

integration_tests_bn! {
    abc_chain_a0: 0.2_f64,
    abc_chain_a1: 0.8_f64,
    abc_chain_b0: 0.38_f64,
    abc_chain_b1: 0.62_f64,
    abc_chain_c0: 0.348_f64,
    abc_chain_c1: 0.652_f64,
    two_parents_p1_true: 0.2_f64,
    two_parents_p1_false: 0.8_f64,
    two_parents_p2_true: 0.6_f64,
    two_parents_p2_false: 0.4_f64,
    two_parents_child_true: 0.396_f64,
    two_parents_child_false: 0.604_f64,
    two_parents_great_children_d_true: 0.6792_f64,
    two_parents_great_children_d_false: 0.3208_f64,
    two_parents_great_children_e_true: 0.5188_f64,
    two_parents_great_children_e_false: 0.4812_f64,
    asia_xray_true: 0.11029_f64,
    asia_xray_false: 0.88971_f64,
    asia_dyspnea_true: 0.435971_f64,
    asia_dyspnea_false: 0.564029_f64,
}
