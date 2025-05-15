use std::fmt;
use std::path::PathBuf;
use chrono;
use malachite::rational::Rational;
use std::fs::File;
use std::io::Write;
use crate::common::rational;


/// Implements a bunch of statistics that are collected during the search
#[derive(Default)]
pub struct Logger<const B: bool> {
    cache_miss: usize,
    cache_access: usize,
    number_or_nodes: usize,
    number_and_nodes: usize,
    total_and_decompositions: usize,
    number_unsat: usize,
    peak_memory: f32,
    lower_bound: f64,
    upper_bound: f64,
    global_timestamp: chrono::DateTime<chrono::Local>,
    outfile_train: Option<File>,
    outfile_test: Option<File>,
}

impl<const B: bool> Logger<B> {
    pub fn cache_miss(&mut self) {
        if B {
            self.cache_miss += 1;
        }
    }

    pub fn cache_access(&mut self) {
        if B {
            self.cache_access += 1;
        }
    }

    pub fn or_node(&mut self) {
        if B {
            self.number_or_nodes += 1;
        }
    }

    pub fn and_node(&mut self) {
        if B {
            self.number_and_nodes += 1;
        }
    }

    pub fn decomposition(&mut self, number_components: usize) {
        if B {
            self.total_and_decompositions += number_components;
            if number_components > 1 {
                self.and_node();
            }
        }
    }
    
    pub fn unsat(&mut self) {
       if B {
            self.number_unsat += 1;
       } 
    }

    pub fn peak_memory(&mut self, peak_memory: f32) {
        if B {
            self.peak_memory = peak_memory;
        }
    }

    pub fn lower_bound(&mut self, lower_bound: f64) {
        if B {
            self.lower_bound = lower_bound;
        }
    }

    pub fn upper_bound(&mut self, upper_bound: f64) {
        if B {
            self.upper_bound = upper_bound;
        }
    }
    
    pub fn print(&self) {
        if B {
            println!("{}", self);
        }
    }

    pub fn train_init(&mut self, outfolder: Option<&PathBuf>, ndacs_train: usize, ndacs_test: usize) {
        if B {
            self.global_timestamp = chrono::Local::now();

            self.outfile_train = {
                let mut out_train = outfolder.map(|x| File::create(x.join(format!("log_{}.csv", self.global_timestamp.format("%Y%m%d-%H%M%S")))).unwrap());
                let mut output= "".to_string();
                output.push_str("epoch_lr,epsilon,epochs_total_duration,avg_errror,"); //avg_distance,
                for i in 0..ndacs_train {
                    output.push_str(&format!("dac{} epoch_error,", i));
                }
                for i in 0..ndacs_train {
                    output.push_str(&format!("dac{} prediction,", i));
                }
                writeln!(out_train.as_mut().unwrap(), "{}", output).unwrap();
                out_train
            };

            if ndacs_test > 0 {
                self.outfile_test = {
                    let mut out_test = outfolder.map(|x| File::create(x.join(format!("test_{}.csv", self.global_timestamp.format("%Y%m%d-%H%M%S")))).unwrap());
                    let mut output= "".to_string();
                    output.push_str("epsilon,avg_errror,"); //avg_distance,
                    for i in 0..ndacs_test {
                        output.push_str(&format!("dac{} epoch_error,", i));
                    }
                    for i in 0..ndacs_test {
                        output.push_str(&format!("dac{} prediction,", i));
                    }
                    writeln!(out_test.as_mut().unwrap(), "{}", output).unwrap();
                    out_test
                };
            }
        }
    }

    pub fn start_train(&mut self) {
        if B {
            self.global_timestamp = chrono::Local::now();
        }
    }

    pub fn log_epoch(&mut self, loss:&[Rational], lr: f64, epsilon:f64, predictions:&[Rational]) {
        if B {
            let mut output = String::new();
            let epoch_duration = (chrono::Local::now() - self.global_timestamp).num_seconds();
            output.push_str(&format!("{:.6},{},{},{:.8},", lr, epsilon, epoch_duration, loss.iter().sum::<Rational>() / rational(loss.len())));
            for l in loss.iter() {
                output.push_str(&format!("{:.6},", l));
            }
            for p in predictions.iter() {
                output.push_str(&format!("{:.6},", p));
            }
            writeln!(self.outfile_train.as_mut().unwrap(), "{}", output).unwrap();
        }
    }

    pub fn log_test(&mut self, loss:&[Rational], epsilon:f64, predictions:&[Rational]) {
        if B {
            let mut output = String::new();
            output.push_str(&format!("{},{:.8},", epsilon, loss.iter().sum::<Rational>() / rational(loss.len())));
            for l in loss.iter() {
                output.push_str(&format!("{:.6},", l));
            }
            for p in predictions.iter() {
                output.push_str(&format!("{:.6},", p));
            }
            writeln!(self.outfile_test.as_mut().unwrap(), "{}", output).unwrap();
        }
    }

}

impl<const B: bool> fmt::Display for Logger<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if B {
            let cache_hit_percentages = 100f64 - (self.cache_miss as f64 / self.cache_access as f64) * 100.0;
            let avg_decomposition = if self.number_and_nodes > 1 {
                (self.total_and_decompositions as f64) / (self.number_and_nodes as f64)
            } else {
                1.0
            };
            writeln!(f,
                "lower bound {} | upper bound {} | cache_hit {:.3} | OR nodes {} | AND nodes {} | avg decomposition {} | #UNSAT {} | Peak memory usage {} Mb",
                self.lower_bound,
                self.upper_bound,
                cache_hit_percentages,
                self.number_or_nodes,
                self.number_and_nodes,
                avg_decomposition,
                self.number_unsat,
                self.peak_memory)
        } else {
            write!(f, "")
        }
    }
}
