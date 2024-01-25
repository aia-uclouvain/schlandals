# Installation

In order to use Schlandals, you must have the [Rust toolchain installed](https://www.rust-lang.org/tools/install).
Once this is done, you can install Schlandals either using Cargo or from sources.

## Installing using Cargo

Run the following command
```bash
cargo install schlandals
```

it will install locally the executable. In Unix-based system you should find the executable in `~/.cargo/bin/schlandals`.

## Installing from sources

```bash
git clone git@github.com:aia-uclouvain/schlandals.git && cd schlandals && cargo build --release
```

it will compile the solver, from sources, and place the executable inside `schlandals/target/release/schlandals`.

## Optional features

### Learning with Torch

It is possible to use the learning module of Schlandals with torch tensors. The torch bindings are provided by the [tch-rs crate](https://github.com/LaurentMazare/tch-rs).
It means that for using this feature, you must libtorch installed on your system. We refer to the [documentation of tch-rs](https://github.com/LaurentMazare/tch-rs?tab=readme-ov-file#getting-started) for the set-up of torch and tch-rs.

Once torch is installed, and the appropriate variables set, you can run either
```bash
cargo install schlandals --features tensor
```
or
```bash
cargo build --release --features tensor
```
