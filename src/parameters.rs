use std::time::Instant;
use std::convert::From;
use std::path::PathBuf;
use std::ffi::OsString;
use crate::{Args, Command, Loss, Optimizer};

pub struct Parameters {
    /// Input problem (String or File)
    input: OsString,
    /// Evidence (file or string)
    evidence: Option<OsString>,
    /// Memory limit for the solving, in megabytes. When reached, the cache is cleared. Note that
    /// this parameter should not be used for compilation.
    memory_limit: u64,
    /// Approximation factor
    epsilon: f64,
    /// Time limit for the search
    timeout: u64,
    /// Time at which the solving started
    start: Instant,
    /// Should the solver be run with limited discrepancy search
    lds: bool,
    /// Should the solver approximate sub-problems
    approx_subproblems: bool,
    /// The initial learning rate
    learning_rate: f64,
    /// The number of epochs
    nepochs: usize,
    /// The timeout used for the training loop, in seconds
    learning_timeout: u64,
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
}

impl Parameters {

    pub fn update_from_args(&mut self, args: &Args) {
    }

    pub fn memory_limit(&self) -> u64 {
        self.memory_limit
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn timeout(&self) -> u64 {
        self.timeout
    }

    pub fn start(&self) -> Instant {
        self.start
    }

    pub fn lds(&self) -> bool {
        self.lds
    }

    pub fn approx_subproblems(&self) -> bool {
        self.approx_subproblems
    }

    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn nepochs(&self) -> usize {
        self.nepochs
    }

    pub fn learning_timeout(&self) -> u64 {
        self.learning_timeout
    }

    pub fn loss(&self) -> Loss {
        self.loss
    }

    pub fn optimizer(&self) -> Optimizer {
        self.optimizer
    }

    pub fn lr_drop(&self) -> f64 {
        self.lr_drop
    }

    pub fn epoch_drop(&self) -> usize {
        self.epoch_drop
    }

    pub fn early_stop_threshold(&self) -> f64 {
        self.early_stop_threshold
    }

    pub fn early_stop_delta(&self) -> f64 {
        self.early_stop_delta
    }

    pub fn patience(&self) -> usize {
        self.patience
    }

    pub fn recompile(&self) -> bool {
        self.recompile
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
                memory_limit: u64::MAX,
                epsilon: 0.0,
                timeout: u64::MAX,
                start: Instant::now(),
                lds: false,
                approx_subproblems: false,
                learning_rate: 0.3,
                nepochs: 6000,
                learning_timeout: u64::MAX,
                loss: Loss::MAE,
                optimizer: Optimizer::Adam,
                lr_drop: 0.75,
                epoch_drop: 100,
                early_stop_threshold: 0.0001,
                early_stop_delta: 0.00001,
                patience: 5,
                recompile: false,
        }
    }
}

impl From<Args> for Parameters {
    fn from(item: Args) -> Self {
        let parameters = Self {
            memory_limit: item.memory,
            epsilon: item.epsilon,
            timeout: item.timeout,
            start: Instant::now(),
            lds: item.lds,
            approx_subproblems: item.approx_subproblems,
            learning_rate: 0.3,
            nepochs: 6000,
            learning_timeout: u64::MAX,
            loss: Loss::MAE,
            optimizer: Optimizer::Adam,
            lr_drop: 0.75,
            epoch_drop: 100,
            early_stop_threshold: 0.0001,
            early_stop_delta: 0.00001,
            patience: 5,
            recompile: false,
        };
        if let Some(v) = item.subcommand {
            match v {
                Command::Compile { fdac, dotfile } => {
                },
                Command::Learn { trainfile, testfile, outfolder, lr, nepochs, ltimeout, loss, jobs, optimizer, lr_drop, epoch_drop, early_stop_threshold, early_stop_delta, patience, equal_init, recompile, e_weighted } => {
                },
            }
        }
        parameters
    }
}
