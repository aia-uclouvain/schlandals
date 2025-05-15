use schlandals::args::Args;
use clap::Parser;

fn main() {
    let args = Args::parse();
    if args.learning() {
        schlandals::learn(args);
    } else {
        schlandals::solve(args);
    }
}
