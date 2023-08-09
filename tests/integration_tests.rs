#![allow(non_snake_case)]
use rug::Float;
use schlandals::*;
use schlandals::Branching::*;

use std::path::PathBuf;
use tempfile::Builder;
use std::io::Write;

use paste::paste;

macro_rules! test_input_with_branching {
    ($dir:ident, $name:ident, $value:expr, $b:ident) => {
        paste!{
            #[test]
            fn [<search_ $b _ $name>]() {
                let filename = format!("tests/instances/{}/{}.cnf", stringify!($dir), stringify!($name));
                let sol = search(PathBuf::from(filename), Branching::$b, false, None).unwrap();
                let expected = Float::with_val(113, $value);
                assert!((expected - sol).abs() < 0.000001);
            }
            
            #[test]
            fn [<compile_ $b _ $name>]() {
                let filename = format!("tests/instances/{}/{}.cnf", stringify!($dir), stringify!($name));
                let sol = compile(PathBuf::from(filename), $b, None, None).unwrap().evaluate();
                let expected = Float::with_val(113, $value);
                assert!((expected - sol).abs() < 0.000001);
            }

            #[test]
            fn [<compile_from_file_ $b _ $name>]() {
                let filename = format!("tests/instances/{}/{}.cnf", stringify!($dir), stringify!($name));
                let dac = compile(PathBuf::from(filename), $b, None, None).unwrap();
                let mut file = Builder::new().prefix("tmp").suffix(".dac").tempfile().unwrap();
                writeln!(file, "{}", dac).unwrap();
                let mut read_dac = read_compiled(PathBuf::from(file.path()), None);
                let sol = read_dac.evaluate();
                let expected = Float::with_val(113, $value);
                assert!((expected - sol).abs() < 0.000001);
            }
        }
    }
}

fn compare_approximate(sol: Float, expected: Float, epsilon: f64) {
    let lb = ((expected.clone()) / (1.0 + epsilon)) - 0.000001;
    let ub = (expected * (1.0 + epsilon)) + 0.000001;
    assert!(lb <= sol && sol <= ub);
}

macro_rules! test_approximate_input_with_branching {
    ($dir:ident, $name:ident, $value:expr, $b:ident) => {
        paste! {
            #[test]
            fn [<approximate_search_0_ $b _ $name>]() {
                let epsilon = 0.0;
                let filename = format!("tests/instances/{}/{}.cnf", stringify!($dir), stringify!($name));
                let sol = approximate_search(PathBuf::from(filename), Branching::$b, false, None, epsilon).unwrap();
                let expected = Float::with_val(113, $value);
                compare_approximate(sol, expected, epsilon);
            }

            #[test]
            fn [<approximate_search_5_ $b _ $name>]() {
                let epsilon = 0.05;
                let filename = format!("tests/instances/{}/{}.cnf", stringify!($dir), stringify!($name));
                let sol = approximate_search(PathBuf::from(filename), Branching::$b, false, None, epsilon).unwrap();
                let expected = Float::with_val(113, $value);
                compare_approximate(sol, expected, epsilon);
            }

            #[test]
            fn [<approximate_search_20_ $b _ $name>]() {
                let epsilon = 0.2;
                let filename = format!("tests/instances/{}/{}.cnf", stringify!($dir), stringify!($name));
                let sol = approximate_search(PathBuf::from(filename), Branching::$b, false, None, epsilon).unwrap();
                let expected = Float::with_val(113, $value);
                compare_approximate(sol, expected, epsilon);
            }

            #[test]
            fn [<approximate_search_50_ $b _ $name>]() {
                let epsilon = 0.5;
                let filename = format!("tests/instances/{}/{}.cnf", stringify!($dir), stringify!($name));
                let sol = approximate_search(PathBuf::from(filename), Branching::$b, false, None, epsilon).unwrap();
                let expected = Float::with_val(113, $value);
                compare_approximate(sol, expected, epsilon);
            }

            #[test]
            fn [<approximate_search_100_ $b _ $name>]() {
                let epsilon = 1.0;
                let filename = format!("tests/instances/{}/{}.cnf", stringify!($dir), stringify!($name));
                let sol = approximate_search(PathBuf::from(filename), Branching::$b, false, None, epsilon).unwrap();
                let expected = Float::with_val(113, $value);
                compare_approximate(sol, expected, epsilon);
            }
        }
    }
}

macro_rules! integration_tests {
    ($dir:ident, $($name:ident: $value:expr,)*) => {
        $(
            test_input_with_branching!{$dir, $name, $value, MinInDegree}
            test_input_with_branching!{$dir, $name, $value, MinOutDegree}
            test_input_with_branching!{$dir, $name, $value, MaxDegree}
        )*
    }
}

macro_rules! integration_tests_approximate {
    ($dir:ident, $($name:ident: $value:expr,)*) => {
        $(
            test_approximate_input_with_branching!{$dir, $name, $value, MinInDegree}
            test_approximate_input_with_branching!{$dir, $name, $value, MinOutDegree}
            test_approximate_input_with_branching!{$dir, $name, $value, MaxDegree}
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

integration_tests_approximate! {
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