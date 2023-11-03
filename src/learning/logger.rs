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

use std::fmt;
use chrono;

/// Implements a bunch of statistics that are collected during the search
#[derive(Default)]
#[cfg(not(tarpaulin_include))]
pub struct Logger<const B: bool> {
    epoch_duration: Vec<i64>,
    epoch_error: Vec<f64>,
    epoch_distance: Vec<f64>,
    epoch_lr: Vec<f64>,
    global_timestamp: chrono::DateTime<chrono::Local>,
}

impl<const B: bool> Logger<B> {
    pub fn start(&mut self) {
        if B {
            if self.epoch_duration.len() == 0 {
                self.global_timestamp = chrono::Local::now();
            }
        }
    }
    pub fn add_epoch(&mut self, loss:f64, expected_distribution: &Vec<Vec<f64>>, predicted_distribution: &Vec<Vec<f64>>, lr: f64) {
        if B {
            self.epoch_duration.push((chrono::Local::now() - self.global_timestamp).num_seconds());
            self.epoch_error.push(loss);
            self.epoch_distance.push(Self::distance(&expected_distribution, &predicted_distribution));
            self.epoch_lr.push(lr);
        }
    }
    fn distance(expected_distribution: &Vec<Vec<f64>>, predicted_distribution: &Vec<Vec<f64>>) -> f64 {
        let mut total = 0.0;
        let mut nb_var = 0;
        for i in 0..expected_distribution.len() {
            nb_var += expected_distribution[i].len();
            for j in 0..expected_distribution[i].len() {
                total += (expected_distribution[i][j] - predicted_distribution[i][j]).abs();
            }
        }
        total /= nb_var as f64;
        total
    }
}

impl<const B: bool> fmt::Display for Logger<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if B {
            let mut output= "epoch_error, epoch_distance, epoch_lr, epochs_total_duration, date\n".to_string();
            for i in 0..self.epoch_error.len() {
                output.push_str(&format!("{}, {}, {}, {}, {}\n", self.epoch_error[i], self.epoch_distance[i], self.epoch_lr[i], self.epoch_duration[i], self.global_timestamp.format("%Y%m%d-%H%M%S")));
            }
            writeln!(f, "{}", output)
        } else {
            write!(f, "")
        }
    }
}
