use clap::Parser;

use std::path::PathBuf;
use std::ffi::OsString;
use crate::{Branching, Caching, Loss, Optimizer};

#[derive(Parser)]
#[clap(name="Schlandals", version, author, about)]
pub struct Args {
    /// The input file
    #[clap(short, long, value_parser)]
    input: PathBuf,
    /// The query, either a string or a file
    #[clap(long, required=false)]
    query: Option<OsString>,
    /// Stops the search/compilation after timeout seconds
    #[clap(short, long, default_value_t=u64::MAX)]
    timeout: u64,
    /// Distribution selection heuristic
    #[clap(short, long, value_enum, default_value_t=Branching::MinInDegree)]
    branching: Branching,
    /// Caching strategy
    #[clap(short, long, value_enum, default_value_t=Caching::Hybrid)]
    caching: Caching,
    /// If present, launch the solver in learning mode
    #[clap(short, long, default_value_t=false)]
    learning: bool,
    /// If present, launch the solver in compilation mode
    #[clap(short, long, default_value_t=false)]
    compile: bool,
    /// Collect stats during the search
    #[clap(long, action)]
    statistics: bool,
    /// The memory limit, in mega-bytes
    #[clap(short, long, default_value_t=u64::MAX)]
    memory: u64,
    /// Epsilon, the quality of the approximation (must be between greater or equal to 0). If 0 or absent, performs exact search
    #[clap(short, long, default_value_t=0.0)]
    epsilon: f64,
    /// Should the solver run lds?
    #[clap(long, default_value_t=false)]
    lds: bool,
    /// Should the solver approximate sub-problems ?
    #[clap(long, default_value_t=false)]
    approx_subproblems: bool,
    /// If the problem is compiled, store a DOT graphical representation in this file
    #[clap(long)]
    dotfile: Option<PathBuf>,
    /// Optional input file for the test set when learning
    #[clap(long, value_parser, value_delimiter=' ')]
    testfile: Option<PathBuf>,
    /// If present, folder in which to store the output files of the learning
    #[clap(long)]
    outfolder: Option<PathBuf>,
    /// Learning rate
    #[clap(short, long, default_value_t=0.3)]
    lr: f64,
    /// Number of epochs
    #[clap(long, default_value_t=6000)]
    nepochs: usize,
    /// If present, define the learning timeout
    #[clap(long, default_value_t=u64::MAX)]
    ltimeout: u64,
    /// Loss to use for the training, default is the MAE
    /// Possible values: MAE, MSE
    #[clap(long, default_value_t=Loss::MAE, value_enum)]
    loss: Loss, 
    /// Number of threads to use for the evaluation of the DACs
    #[clap(long, default_value_t=1, short)]
    jobs: usize,
    /// The optimizer to use if `tensor` is selected as semiring
    #[clap(long, short, default_value_t=Optimizer::Adam, value_enum)]
    optimizer: Optimizer,
    /// The drop in the learning rate to apply at each step
    #[clap(long, default_value_t=0.75)]
    lr_drop: f64,
    /// The number of epochs after which to drop the learning rate
    /// (i.e. the learning rate is multiplied by `lr_drop`)
    #[clap(long, default_value_t=100)]
    epoch_drop: usize,
    /// The stopping criterion for the training
    /// (i.e. if the loss is below this value, stop the training)
    #[clap(long, default_value_t=0.0001)]
    early_stop_threshold: f64,
    /// The minimum of improvement in the loss to consider that the training is still improving
    /// (i.e. if the loss is below this value for a number of epochs, stop the training)
    #[clap(long, default_value_t=0.00001)]
    early_stop_delta: f64,
    /// The number of epochs to wait before stopping the training if the loss is not improving
    /// (i.e. if the loss is below this value for a number of epochs, stop the training)
    #[clap(long, default_value_t=5)]
    patience: usize,
    /// If present, initialize the distribution weights as 1/|d|, |d| being the number of values for the distribution
    #[clap(long, action)]
    equal_init: bool,
    /// If present, recompile the circuits at each epoch
    #[clap(long, action)]
    recompile: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            input: PathBuf::default(),
            query: None,
            timeout: u64::MAX,
            branching: Branching::MinInDegree,
            caching: Caching::Hybrid,
            learning: false,
            compile: false,
            statistics: false,
            memory: u64::MAX,
            epsilon: 0.0,
            lds: false,
            approx_subproblems: false,
            dotfile: None,
            testfile: None,
            outfolder: None,
            lr: 0.3,
            nepochs: 6000,
            ltimeout: u64::MAX,
            loss: Loss::MAE,
            jobs: 1,
            optimizer: Optimizer::Adam,
            lr_drop: 0.75,
            epoch_drop:100,
            early_stop_threshold: 0.0001,
            early_stop_delta: 0.00001,
            patience: 5,
            equal_init: false,
            recompile: false,
        }
    }
}

impl Args {

    pub fn input(&self) -> &PathBuf {
        &self.input
    }

    pub fn query(&self) -> Option<OsString> {
        self.query.clone()
    }

    pub fn timeout(&self) -> u64 {
        self.timeout
    }

    pub fn branching(&self) -> Branching {
        self.branching
    }

    pub fn caching(&self) -> Caching {
        self.caching
    }

    pub fn learning(&self) -> bool {
        self.learning
    }

    pub fn compile(&self) -> bool {
        self.compile
    }

    pub fn statistics(&self) -> bool {
        self.statistics
    }

    pub fn memory(&self) -> u64 {
        self.memory
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn lds(&self) -> bool {
        self.lds
    }

    pub fn approx_subproblems(&self) -> bool {
        self.approx_subproblems
    }

    pub fn dotfile(&self) -> Option<PathBuf> {
        self.dotfile.clone()
    }

    pub fn testfile(&self) -> Option<PathBuf> {
        self.testfile.clone()
    }

    pub fn outfolder(&self) -> Option<PathBuf> {
        self.outfolder.clone()
    }

    pub fn lr(&self) -> f64 {
        self.lr
    }

    pub fn nepochs(&self) -> usize {
        self.nepochs
    }

    pub fn ltimeout(&self) -> u64 {
        self.ltimeout
    }

    pub fn loss(&self) -> Loss {
        self.loss
    }

    pub fn jobs(&self) -> usize {
        self.jobs
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

    pub fn equal_init(&self) -> bool {
        self.equal_init
    }

    pub fn recompile(&self) -> bool {
        self.recompile
    }

    pub fn set_input(&mut self, value: PathBuf) {
        self.input = value;
    }

    pub fn set_query(&mut self, value: Option<OsString>) {
        self.query = value;
    }

    pub fn set_timeout(&mut self, value: u64) {
        self.timeout = value;
    }

    pub fn set_branching(&mut self, value: Branching) {
        self.branching = value;
    }

    pub fn set_caching(&mut self, value: Caching) {
        self.caching = value;
    }

    pub fn set_learning(&mut self, value: bool) {
        self.learning = value;
    }

    pub fn set_compile(&mut self, value: bool) {
        self.compile = value;
    }

    pub fn set_statistics(&mut self, value: bool) {
        self.statistics = value;
    }

    pub fn set_memory(&mut self, value: u64) {
        self.memory = value;
    }

    pub fn set_epsilon(&mut self, value: f64) {
        self.epsilon = value;
    }

    pub fn set_lds(&mut self, value: bool) {
        self.lds = value;
    }

    pub fn set_approx_subproblems(&mut self, value: bool) {
        self.approx_subproblems = value;
    }

    pub fn set_dotfile(&mut self, value: Option<PathBuf>) {
        self.dotfile = value;
    }

    pub fn set_testfile(&mut self, value: Option<PathBuf>) {
        self.testfile = value;
    }

    pub fn set_outfolder(&mut self, value: Option<PathBuf>) {
        self.outfolder = value;
    }

    pub fn set_lr(&mut self, value: f64) {
        self.lr = value;
    }

    pub fn set_nepochs(&mut self, value: usize) {
        self.nepochs = value;
    }

    pub fn set_ltimeout(&mut self, value: u64) {
        self.ltimeout = value;
    }

    pub fn set_loss(&mut self, value: Loss) {
        self.loss = value;
    }

    pub fn set_jobs(&mut self, value: usize) {
        self.jobs = value;
    }

    pub fn set_optimizer(&mut self, value: Optimizer) {
        self.optimizer = value;
    }

    pub fn set_lr_drop(&mut self, value: f64) {
        self.lr_drop = value;
    }

    pub fn set_epoch_drop(&mut self, value: usize) {
        self.epoch_drop = value;
    }

    pub fn set_early_stop_threshold(&mut self, value: f64) {
        self.early_stop_threshold = value;
    }

    pub fn set_early_stop_delta(&mut self, value: f64) {
        self.early_stop_delta = value;
    }

    pub fn set_patience(&mut self, value: usize) {
        self.patience = value;
    }

    pub fn set_equal_init(&mut self, value: bool) {
        self.equal_init = value;
    }

    pub fn set_recompile(&mut self, value: bool) {
        self.recompile = value;
    }
}
