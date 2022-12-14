use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use schlandals::branching::*;
use schlandals::components::ComponentExtractor;
use schlandals::ppidimacs::graph_from_ppidimacs;
use schlandals::solver::QuietSolver;
use schlandals::trail::StateManager;

use std::path::PathBuf;


macro_rules! set_up_solvers {
    ($($name:ident: [$instance:expr, $b:ident, $c:ident],)*) => {
        $(
            let filename = format!("benches/instances/{}.ppidimacs", $instance);
            let path = PathBuf::from(filename);
            let mut state = StateManager::default();
            let (g, _v) = graph_from_ppidimacs(&path, &mut state).unwrap();
            let component_extractor = ComponentExtractor::new(&g, &mut state);
            let mut branching_heuristic = $b::default();
            let mut solver = QuietSolver::new(g, state, component_extractor, &mut branching_heuristic);
            $c.bench_function($instance, |b| b.iter(|| solver.solve()));
        )*
    }
}

macro_rules! make_benches {
    ($($instance:expr,)*) => {
        pub fn bench(c: &mut Criterion) {
            $(
                set_up_solvers! {
                    fiedler_nb_diff: [$instance,  NeighborDiffFiedler, c],
                }
            )*
        }
        criterion_group!(benches, bench);
        criterion_main!(benches);
    }
}

make_benches! {
    "bn/asia/dysp_no",
    "bn/asia/dysp_yes",
    "bn/asia/xray_no",
    "bn/asia/xray_yes",
    "bn/cancer/Dyspnoea_False",
    "bn/cancer/Dyspnoea_True",
    "bn/cancer/Xray_negative",
    "bn/cancer/Xray_positive",
    "bn/earthquake/JohnCalls_False",
    "bn/earthquake/JohnCalls_True",
    "bn/earthquake/MaryCalls_False",
    "bn/earthquake/MaryCalls_True",
    "bn/survey/T_car",
    "bn/survey/T_other",
    "bn/survey/T_train",
    "pg/Albania/2_5",
    "pg/Albania/2_6",
    "pg/Albania/7_5",
    "pg/Albania/8_4",
    "pg/Albania/9_1",
    "pg/Albania/9_8",
    "pg/Albania/10_3",
    "pg/Albania/11_13",
    "pg/Albania/13_4",
    "pg/Albania/15_2",
    "pg/Armenia/0_4",
    "pg/Armenia/0_5",
    "pg/Armenia/1_3",
    "pg/Armenia/2_0",
    "pg/Armenia/3_4",
    "pg/Armenia/4_2",
    "pg/Armenia/5_4",
    "pg/Croatia/0_15",
    "pg/Croatia/2_19",
    "pg/Croatia/3_20",
    "pg/Croatia/7_8",
    "pg/Croatia/18_5",
    "pg/Croatia/20_13",
    "pg/Croatia/21_4",
    "pg/Croatia/22_19",
    "pg/Croatia/23_2",
    "pg/Delaware/3_0",
    "pg/Delaware/3_1",
    "pg/Delaware/4_2",
    "pg/Delaware/6_12",
    "pg/Delaware/7_0",
    "pg/Delaware/7_3",
    "pg/Delaware/9_14",
    "pg/Delaware/10_12",
    "pg/Delaware/13_6",
    "pg/Delaware/13_12",
}