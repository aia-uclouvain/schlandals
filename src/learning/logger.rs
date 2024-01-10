//Schlandals
//Copyright (C) 2022-2023 A. Dubray
//
//This program is free software: you can redistribute it and/or modify
//it under the terms of the GNU Affero General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU Affero General Public License for more details.
//
//You should have received a copy of the GNU Affero General Public License
//along with this program.  If not, see <http://www.gnu.org/licenses/>.

use std::path::PathBuf;
use chrono;
//use rug::Float;
use std::fs::File;
use std::io::Write;

/// Implements a bunch of statistics that are collected during the search
#[cfg(not(tarpaulin_include))]
pub struct Logger<const B: bool> {
    global_timestamp: chrono::DateTime<chrono::Local>,
    outfile: Option<File>,
}

impl<const B: bool> Logger<B> {
    pub fn default() -> Self {
        Self {
            global_timestamp: chrono::Local::now(),
            outfile: None,
        }
    }
    pub fn new(outfolder:Option<&PathBuf>, ndacs:usize) -> Self {
        let global_timestamp = chrono::Local::now();
        let csv_file = if B{
            let mut out = match outfolder {
                Some(x) => {
                    Some(File::create(x.join(format!("log_{}.csv", global_timestamp.format("%Y%m%d-%H%M%S")))).unwrap())
                },
                None => None,
            };
            let mut output= "".to_string();
            output.push_str("epoch_lr,epsilon,epochs_total_duration,avg_errror,"); //avg_distance,
            for i in 0..ndacs {
                output.push_str(&format!("dac{} epoch_error,", i));
            }
            for i in 0..ndacs {
                output.push_str(&format!("dac{} prediction,", i));
            }
            /* for i in 0..distributions.len() {
                for j in 0..distributions[i].len() {
                    output.push_str(&format!("distribution{}_{} epoch_distance,",i, j));
                }
            } */
            writeln!(out.as_mut().unwrap(), "{}", output).unwrap();
            out
        } else {
            None
        };
        Self {
            global_timestamp,
            outfile: csv_file,
        }
    }
    pub fn start(&mut self) {
        if B {
            self.global_timestamp = chrono::Local::now();
        }
    }
    pub fn log_epoch(&mut self, loss:&Vec<f64>, lr: f64, epsilon:f64, predictions:&Vec<f64>) { //expected_distribution: &Vec<Vec<f64>>, predicted_distribution: &Vec<Vec<f64>>, gradients: &Vec<Vec<Float>>,
        if B {
            let mut output = String::new();
            let epoch_duration = (chrono::Local::now() - self.global_timestamp).num_seconds();
            //let distances = Self::distance(&expected_distribution, &predicted_distribution, &gradients);
            //let non_null_distances: Vec<f64> = distances.iter().flatten().filter(|d| **d!=0.0).copied().collect();
            output.push_str(&format!("{:.6},{},{},{:.8},", lr, epsilon, epoch_duration, loss.iter().sum::<f64>() / loss.len() as f64)); //, non_null_distances.iter().sum::<f64>() / non_null_distances.iter().count() as f64
            for l in loss.iter() {
                output.push_str(&format!("{:.6},", l));
            }
            for p in predictions.iter() {
                output.push_str(&format!("{:.6},", p));
            }
            /* for distr in distances.iter() {
                for d in distr.iter() {
                    output.push_str(&format!("{:.6},", d));
                }
            } */
            writeln!(self.outfile.as_mut().unwrap(), "{}", output).unwrap();
            
        }
    }
    /* fn distance(expected_distribution: &Vec<Vec<f64>>, predicted_distribution: &Vec<Vec<f64>>, gradients: &Vec<Vec<Float>>) -> Vec<Vec<f64>> {
        let mut total: Vec<Vec<f64>> = vec![];
        for i in 0..expected_distribution.len() {
            let mut tmp: Vec<f64> = vec![];
            for j in 0..expected_distribution[i].len() {
                if gradients[i][j] != 0.0 {
                    tmp.push((expected_distribution[i][j] - predicted_distribution[i][j]).abs());
                }
                else{
                    tmp.push(0.0);
                }
            }
            total.push(tmp);
        }
        total
    } */
}
