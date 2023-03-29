#![allow(non_snake_case)]
use rug::Float;
use schlandals;
use schlandals::branching::*;
use schlandals::propagator::FTReachablePropagator;
use schlandals::components::*;
use schlandals::ppidimacs::graph_from_ppidimacs;
use schlandals::solver::QuietSolver;
use search_trail::StateManager;

use std::path::PathBuf;


macro_rules! integration_tests {
    ($dir:ident, $($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let filename = format!("tests/instances/{}/{}.ppidimacs", stringify!($dir), stringify!($name));
                let mut state = StateManager::default();
                let mut propagator = FTReachablePropagator::default();
                let path = PathBuf::from(filename);
                let graph = graph_from_ppidimacs(&path, &mut state, &mut propagator);
                let component_extractor = ComponentExtractor::new(&graph, &mut state);
                let mut branching_heuristic = Fiedler::default();
                let mut solver = QuietSolver::new(graph, state, component_extractor, &mut branching_heuristic, propagator, 1000);
                let sol = solver.solve().unwrap();
                let expected = Float::with_val(113, $value);
                println!("Expected {:}, actual {:?}", expected, sol);
                assert!((expected - sol).abs() < 0.000001);
            }
        )*
    }
}

integration_tests! {
    bayesian_networks,
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

integration_tests! {
    water_supply_network,
    Net1_2_23: 1.0 - 0.826934,
    Net1_2_32: 1.0 - 0.669921,
    Net1_9_23: 1.0 - 0.710383,
    Net1_9_32: 1.0 - 0.710383,
    Net3_1_219: 1.0 - 0.134933,
    Net3_1_225: 1.0 - 0.134933,
    Net3_1_231: 1.0 - 0.190353,
    Net3_1_243: 1.0 - 0.134933,
    Net3_1_251: 1.0 - 0.162173,
    Net3_1_253: 1.0 - 0.103308,
    Net3_2_251: 1.0 - 0.669921,
    Net3_2_253: 1.0 - 0.669921,
    Net3_3_131: 1.0 - 0.586181,
    Net3_3_141: 1.0 - 0.512908,
    Net3_15_141: 1.0 - 0.765625,
    Net3_35_203: 1.0 - 0.392695,
    Net3_35_219: 1.0 - 0.216378,
    Net3_35_225: 1.0 - 0.216378,
    Net3_35_231: 1.0 - 0.305247,
    Net3_35_243: 1.0 - 0.216378,
    Net3_35_251: 1.0 - 0.260059,
    Net3_35_253: 1.0 - 0.165664,
    Net3_147_141: 1.0 - 0.765625,
    Net3_147_153: 1.0 - 0.669921,
    Net3_167_203: 1.0 - 0.491746,
    Net3_167_219: 1.0 - 0.247413,
    Net3_167_225: 1.0 - 0.247413,
    Net3_167_231: 1.0 - 0.349029,
    Net3_167_243: 1.0 - 0.247413,
    Net3_167_251: 1.0 - 0.297359,
    Net3_167_253: 1.0 - 0.189425,
}