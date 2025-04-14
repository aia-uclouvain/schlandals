use malachite::rational::Rational;
use malachite::base::num::arithmetic::traits::{Abs, Pow};

use crate::{Loss, Optimizer};
use crate::common::rational;

pub mod learner;
mod logger;

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
    early_stop_threshold: Rational,
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
        Self {lr, nepochs, compilation_timeout, learn_timeout, loss, optimizer, lr_drop, epoch_drop, early_stop_threshold: rational(early_stop_threshold), early_stop_delta, 
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
    pub fn early_stop_threshold(&self) -> &Rational {
        &self.early_stop_threshold
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
    fn loss(&self, predicted: &Rational, expected: &Rational) -> Rational;
    /// Evaluates the gradient of the loss given a predicted and expected output
    fn gradient(&self, predicted: &Rational, expected: &Rational) -> Rational;
}

impl LossFunctions for Loss {

    fn loss(&self, predicted: &Rational, expected: &Rational) -> Rational {
        match self {
            Loss::MSE => (predicted - expected).pow(2i64),
            Loss::MAE => (predicted - expected).abs(),
        }
    }

    fn gradient(&self, predicted: &Rational, expected: &Rational) -> Rational {
        match self {
            Loss::MSE => rational(2.0) * (predicted - expected),
            Loss::MAE => {
                if predicted == expected {
                    rational(0.0)
                } else if predicted > expected {
                    rational(1.0)
                } else {
                    rational(-1.0)
                }
            },
        }
    }
}
