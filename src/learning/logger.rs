use std::path::PathBuf;
use chrono;
use malachite::Rational;
use std::fs::File;
use std::io::Write;
use crate::common::F128;

/// Implements a bunch of statistics that are collected during the search
pub struct Logger<const B: bool> {
    global_timestamp: chrono::DateTime<chrono::Local>,
    outfile_train: Option<File>,
    outfile_test: Option<File>,
}

impl<const B: bool> Logger<B> {
    pub fn new(outfolder:Option<&PathBuf>, ndacs_train:usize, ndacs_test:usize) -> Self {
        let global_timestamp = chrono::Local::now();

        let outfile_train = if B{
            let mut out_train = outfolder.map(|x| File::create(x.join(format!("log_{}.csv", global_timestamp.format("%Y%m%d-%H%M%S")))).unwrap());
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
        } else {
            None
        };

        let outfile_test = if B && ndacs_test > 0 {
            let mut out_test = outfolder.map(|x| File::create(x.join(format!("test_{}.csv", global_timestamp.format("%Y%m%d-%H%M%S")))).unwrap());
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
        } else {
            None
        };

        Self {
            global_timestamp,
            outfile_train,
            outfile_test,
        }
    }
    pub fn start(&mut self) {
        if B {
            self.global_timestamp = chrono::Local::now();
        }
    }
    pub fn log_epoch(&mut self, loss:&[Rational], lr: f64, epsilon:f64, predictions:&[Rational]) {
        if B {
            let mut output = String::new();
            let epoch_duration = (chrono::Local::now() - self.global_timestamp).num_seconds();
            output.push_str(&format!("{:.6},{},{},{:.8},", lr, epsilon, epoch_duration, loss.iter().sum::<Rational>() / F128!(loss.len())));
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
            output.push_str(&format!("{},{:.8},", epsilon, loss.iter().sum::<Rational>() / F128!(loss.len())));
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
