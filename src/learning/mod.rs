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

use crate::Loss;

pub mod learner;
#[cfg(feature = "tensor")]
pub mod tensor_learner;
mod utils;
mod logger;

pub trait Learning {
    fn train(&mut self, nepochs: usize, init_lr: f64, loss: Loss, timeout: u64);
}

/// This trait provide multiple functions to use on loss functions. In particular, every loss
/// function must implements
///     - Their computation (loss)
///     - Their gradient (gradient)
pub trait LossFunctions {
    /// Evaluates the loss given a predicted and expected output
    fn loss(&self, predicted: f64, expected: f64) -> f64;
    /// Evaluates the gradient of the loss given a predicted and expected output
    fn gradient(&self, predicted: f64, expected: f64) -> f64;
}

impl LossFunctions for Loss {

    fn loss(&self, predicted: f64, expected: f64) -> f64 {
        match self {
            Loss::MSE => (predicted - expected).powi(2),
            Loss::MAE => (predicted - expected).abs(),
        }
    }

    fn gradient(&self, predicted: f64, expected: f64) -> f64 {
        match self {
            Loss::MSE => 2.0 * (predicted - expected),
            Loss::MAE => {
                if predicted == expected {
                    0.0
                } else if predicted > expected {
                    1.0
                } else {
                    -1.0
                }
            },
        }
    }
}
