//Schlandals
//Copyright (C) 2022 A. Dubray, L. Dierckx
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

use self::learner::Learner;
use rug::Float;
use tch::Tensor;
use crate::Branching;
use crate::Semiring;
use crate::Loss;

pub mod learner;
mod logger;

pub enum Learn {
    ProbLog(Learner<Float, true>),
    ProbQuiet(Learner<Float, false>),
    TensorLog(Learner<Tensor, true>),
    TensorQuiet(Learner<Tensor, false>),
}

impl Learn {

    pub fn new(inputs: Vec<PathBuf>, expected: Vec<f64>, epsilon: f64, branching: Branching, outfolder: Option<PathBuf>, jobs: usize, do_log: bool, semiring: Semiring) -> Self {
        match semiring {
            Semiring::Probability => {
                if do_log {
                    Learn::ProbLog(Learner::<Float, true>::new(inputs, expected, epsilon, branching, outfolder, jobs))
                } else {
                    Learn::ProbQuiet(Learner::<Float, false>::new(inputs, expected, epsilon, branching, outfolder, jobs))
                }
            },
            Semiring::Tensor => {
                if do_log {
                    Learn::TensorLog(Learner::<Tensor, true>::new(inputs, expected, epsilon, branching, outfolder, jobs))
                } else {
                    Learn::TensorQuiet(Learner::<Tensor, false>::new(inputs, expected, epsilon, branching, outfolder, jobs))
                }
            }
        }
    }

    pub fn train(&mut self, nepochs:usize, lr: f64, loss: Loss, timeout: i64) {
        match self {
            Learn::ProbLog(ref mut l) => l.train(nepochs, lr, loss, timeout),
            Learn::ProbQuiet(ref mut l) => l.train(nepochs, lr, loss, timeout),
            Learn::TensorLog(ref mut l) => l.train(nepochs, lr, loss, timeout),
            Learn::TensorQuiet(ref mut l) => l.train(nepochs, lr, loss, timeout),
        }
    }
}
