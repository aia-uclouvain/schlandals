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

use crate::{Loss, Optimizer};

pub mod learner;
#[cfg(feature = "tensor")]
pub mod tensor_learner;
mod utils;
mod logger;
use std::path::PathBuf;
use crate::Branching;
use crate::ApproximateMethod;

pub trait Learning {
    fn train(& mut self, params:&LearnParameters, inputs: &Vec<PathBuf>, branching: Branching, approx:ApproximateMethod, compile_timeout: u64);
}

pub struct LearnParameters {
    /// The initial learning rate
    lr: f64,
    /// The number of epochs
    nepochs: usize,
    /// The timeout used for the compilations, in seconds
    compilation_timeout: u64,
    /// The timeout used for the training loop, in seconds
    learn_timeout: u64,
    /// The loss function
    loss: Loss,
    /// The optimizer
    optimizer: Optimizer,
    /// The learning rate decay
    lr_drop: f64,
    /// The number of epochs after which the learning rate is dropped
    epoch_drop: usize,
    /// The error threshold under which the training is stopped
    early_stop_threshold: f64,
    /// The minimum delta between two epochs to consider that the training is still improving
    early_stop_delta: f64,
    /// The number of epochs to wait before stopping the training if the loss is not improving
    patience: usize,
    /// Whether to recompile the circuits at each epoch
    recompile: bool,
    /// Whether to weight the learning with the epsilon value
    e_weighted: bool,
}

impl LearnParameters {
    
    pub fn new(lr: f64, nepochs: usize, compilation_timeout: u64, learn_timeout: u64, loss: Loss, optimizer: Optimizer, lr_drop: f64, epoch_drop: usize, 
               early_stop_threshold: f64, early_stop_delta: f64, patience: usize, recompile:bool, e_weighted:bool) -> Self {
        Self {lr, nepochs, compilation_timeout, learn_timeout, loss, optimizer, lr_drop, epoch_drop, early_stop_threshold, early_stop_delta, 
              patience, recompile, e_weighted}
    }

    /// Returns the learning rate
    pub fn lr(&self) -> f64 {
        self.lr
    }

    /// Returns the maximum number of epochs allowed for the learning
    pub fn nepochs(&self) -> usize {
        self.nepochs
    }

    /// Returns the timeout used for the compilation of the queries (in seconds)
    pub fn compilation_timeout(&self) -> u64 {
        self.compilation_timeout
    }

    /// Return the timeout used for the learning loop (in seconds)
    pub fn learning_timeout(&self) -> u64 {
        self.learn_timeout
    }

    /// Return the loss
    pub fn loss(&self) -> Loss {
        self.loss
    }

    /// Return the optimizer
    pub fn optimizer(&self) -> Optimizer {
        self.optimizer
    }

    /// Returns the decay factor of the learning rate
    pub fn lr_drop(&self) -> f64 {
        self.lr_drop
    }

    /// Returns the frequence (number of epoch) at which the learning rate is decreased
    pub fn epoch_drop(&self) -> usize {
        self.epoch_drop
    }

    /// Returns the loss threshold at which the learning can do an early stopping
    pub fn early_stop_threshold(&self) -> f64 {
        self.early_stop_threshold
    }

    /// Returns the delta for early stopping. If the improvement between two successive epochs (in
    /// terms of loss) is less than this value, the training is considered to be stalling
    pub fn early_stop_delta(&self) -> f64 {
        self.early_stop_delta
    }

    /// Required number non-improved epochs (see `early_stop_delta`) for the early stopping to be
    /// triggered
    pub fn patience(&self) -> usize {
        self.patience
    }

    /// Whether to recompile the circuits at each epoch
    pub fn recompile(&self) -> bool {
        self.recompile
    }

    /// Whether to weight the learning with the epsilon value
    pub fn e_weighted(&self) -> bool {
        self.e_weighted
    }
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
