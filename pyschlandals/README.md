# Installation
pyschlandals is not yet installable from `pip`. But by cloning the main repository of Schlandals, you can install it in your current python environment.
To do so you need to install [the Rust toolchain](https://doc.rust-lang.org/cargo/getting-started/installation.html). Once this is done, you should have `cargo` install.
You can then install [Maturin](https://www.maturin.rs/) with `cargo install maturin`.
Then, from the source directory of `pyschlandals`, run `maturin develop`.

# Usage

The search and compiler main function are exported in python and can be used directly with the input filename and branching heuristic as follows

```python
from pyschlandals.search import exact
from pyschlandals.compiler import compile
from pyschlandals import BranchingHeuristic

filename = '../tests/instances/bayesian_networks/asia_xray_false.cnf'
print(exact(filename, BranchingHeuristic.MinInDegree))

dac = compile(filename, BranchingHeuristic.MinInDegree)
print(dac.get_circuit_probability())
```

Documentation for how to access and update circuits parameters will come soon.
