#![allow(non_snake_case)]
use rug::Float;
use schlandals::*;
use schlandals::Branching::*;
use schlandals::diagrams::dac::dac::Dac;

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
                let sol = search(PathBuf::from(filename), Branching::$b, false, None, 0.0, schlandals::ApproximateMethod::Bounds, u64::MAX, false);
                assert!(($value - sol).abs() < 0.000001);
            }
            
            #[test]
            fn [<compile_ $b _ $name>]() {
                let filename = format!("tests/instances/{}/{}.cnf", stringify!($dir), stringify!($name));
                let sol = compile(PathBuf::from(filename), $b, None, None, 0.0, schlandals::ApproximateMethod::Bounds, u64::MAX);
                assert!(($value - sol).abs() < 0.000001);
            }

            /*
            #[test]
            fn [<compile_from_file_ $b _ $name>]() {i
                let filename = format!("tests/instances/{}/{}.cnf", stringify!($dir), stringify!($name));
                let dac = compile(PathBuf::from(filename), $b, None, None, 0.0).unwrap();
                let mut file = Builder::new().prefix("tmp").suffix(".dac").tempfile().unwrap();
                writeln!(file, "{}", dac).unwrap();
                let mut read_dac: Dac<Float> = Dac::from_file(&PathBuf::from(file.path()));
                let sol = read_dac.evaluate();
                let expected = Float::with_val(113, $value);
                assert!((expected - sol).abs() < 0.000001);
            }
            */
        }
    }
}

macro_rules! integration_tests {
    ($dir:ident, $($name:ident: $value:expr,)*) => {
        $(
            test_input_with_branching!{$dir, $name, $value, MinInDegree}
            //test_input_with_branching!{$dir, $name, $value, MinOutDegree}
            //test_input_with_branching!{$dir, $name, $value, MaxDegree}
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
    child_12:0.298490221753424,
    child_17:0.743480924570115,
    child_10:0.2565046533936,
    child_4:0.4886932367511675,
    child_11:0.24757780881298372,
    child_8:0.15498136570437,
    child_13:0.21606916843522153,
    child_3:0.3714316465155469,
    child_15:0.1546524267497646,
    child_18:0.6489918355500001,
    child_2:0.7133313761225,
    child_20:0.17148510286900004,
    child_5:0.13987511673328576,
    child_7:0.49225850221357303,
    child_6:0.352760132082057,
    child_1:0.28666862387750003,
    child_9:0.7434953466064,
    child_16:0.256519075429885,
    child_14:0.0832103742486062,
    child_19:0.17952306158100004,
    alarm_24:0.0313643748352,
    alarm_22:0.7151850570054399,
    alarm_23:0.06239805721408,
    alarm_4:0.731104,
    alarm_30:0.25382320256,
    alarm_8:0.20693,
    alarm_3:0.11434100000000001,
    alarm_32:0.507944100096,
    alarm_29:0.027214453568,
    alarm_21:0.03469812600832001,
    alarm_18:0.043227342068881416,
    alarm_2:0.9455,
    alarm_20:0.05730683837218202,
    alarm_31:0.211018243776,
    alarm_25:0.19105251094528003,
    alarm_5:0.154555,
    alarm_7:0.6787289999999999,
    alarm_6:0.11434100000000001,
    alarm_26:0.049600000000000005,
    alarm_1:0.05450000000000001,
    alarm_27:0.8929,
    alarm_28:0.0575,
    alarm_19:0.8647676935506166,
    sachs_12:0.05320002410019898,
    sachs_10:0.8400913441873153,
    sachs_4:0.5394062847821939,
    sachs_11:0.1067086317124857,
    sachs_8:0.1441091315703085,
    sachs_3:0.08023205355751407,
    sachs_2:0.31037461849502496,
    sachs_5:0.3827686155448429,
    sachs_7:0.7386286352741057,
    sachs_6:0.07782509967296322,
    sachs_1:0.609393327947461,
    sachs_9:0.11726223315558573,
    earthquake_4:0.978881202,
    earthquake_3:0.021118798,
    earthquake_2:0.93630293,
    earthquake_1:0.06369707000000001,
    andes_12:0.37,
    andes_17:0.09000000000000001,
    andes_10:0.44,
    andes_4:0.98,
    andes_11:0.63,
    andes_30:0.4832,
    andes_8:0.2965226344767294,
    andes_13:0.63,
    andes_3:0.02,
    andes_29:0.5168,
    andes_18:0.91,
    andes_2:0.98,
    andes_5:0.02,
    andes_7:0.7034773655232707,
    andes_6:0.98,
    andes_1:0.02,
    andes_9:0.5599999999999999,
    andes_14:0.37,
    win95pts_24:0.052095066712619535,
    win95pts_22:0.8894582059389688,
    win95pts_12:0.0076,
    win95pts_17:0.8982314,
    win95pts_10:0.0149,
    win95pts_23:0.9479049332873805,
    win95pts_4:0.914375275,
    win95pts_11:0.9924,
    win95pts_30:0.040949990500000005,
    win95pts_8:0.010444999555,
    win95pts_13:0.99161600384,
    win95pts_3:0.08562472500000001,
    win95pts_15:0.7052474034335073,
    win95pts_32:0.107999992,
    win95pts_29:0.9590500095,
    win95pts_21:0.11054179406103123,
    win95pts_18:0.10176860000000001,
    win95pts_31:0.892000008,
    win95pts_25:0.9790400096,
    win95pts_5:0.13778304157445004,
    win95pts_7:0.989555000445,
    win95pts_6:0.86221695842555,
    win95pts_26:0.0209599904,
    win95pts_27:0.9940100099,
    win95pts_9:0.9851,
    win95pts_16:0.2947525965664926,
    win95pts_28:0.0059899901,
    win95pts_14:0.00838399616,
    cancer_4:0.6959295,
    cancer_3:0.3040705,
    cancer_2:0.7918590000000001,
    cancer_1:0.20814100000000002,
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
