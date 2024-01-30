# Installation

Before installing Schlandals, be sure to have [the rust toolchain installed](https://www.rust-lang.org/tools/install).

## From source (recommended)

We assume that the commands are ran from `CURRENT_DIR`
```bash
git clone git@github.com:aia-uclouvain/schlandals.git && cd schlandals && cargo build --release
ln -s $CURRENT_DIR/schlandals/target/release/schlandals $HOME/.local/bin
```
This compiles the code in `release` mode (with optimizations included) and add a symbolic links to the local binaries directory.

## Using Cargo

Note that the code is still actively developed and it should not be expected that the version of the code on crates.io is the most up-to-date.
The main branch of Schlandals should be stable enough for most use cases.

```bash
cargo install schlandals
```
This will put the `schlandals` executable in `~/.cargo/bin/`

## Building Schlandals with Torch support

Schlandals supports [learning parameters](./learning/README.md) using Torch tensors.
We use the [tch-rs crate](https://github.com/LaurentMazare/tch-rs) for the bindings with libtorch tensors and this feature can be installed by
adding `--features tensor` as a flag to the install (`cargo install schlandals --features tensor`) or build (`cargo build --release --features tensor`) commands.
Note that this requires to have libtorch install on your system and some environment variables set. For more information, see [tch install page](https://github.com/LaurentMazare/tch-rs?tab=readme-ov-file#getting-started).
