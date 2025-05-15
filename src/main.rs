use schlandals::Args;
use schlandals::Command;
use clap::Parser;
use schlandals::parameters::*;

fn main() {
    let args = Args::parse();
    match args.subcommand {
        Some(Command::Compile { .. }) => { schlandals::compile(args); },
        Some(Command::Learn { .. }) => { schlandals::learn(args); },
        None => {schlandals::search(args); },
    };
}
