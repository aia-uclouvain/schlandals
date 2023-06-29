![Rust](https://github.com/AlexandreDubray/schlandals/actions/workflows/rust.yml/badge.svg)
[![codecov](https://codecov.io/gh/AlexandreDubray/schlandals/branch/main/graph/badge.svg?token=J4J2I9Q9KX)](https://codecov.io/gh/AlexandreDubray/schlandals)

**The code is still in early development and should not be considered stable**

# schlandals

Schlandals is a projected weighted model counter which targets inference in probabilistic models.
Current known probability queries that can be solved with the solver include
  - Computing the probability of some target variables (given some evidences) in a Bayesian network
  - Computing the probability that two nodes are connected in a probabilistic graph

The solver currently supports two types of solving
  - Search based solving with a DPLL-style backtracing search
  - Compiling into (or read from a file) an AND/OR Multi-valued decision diagram (AOMDD) (still in early development)

# Problem specification

A model counter is a  program that counts the number of satisfying assignments of a boolean formula F.
In its projected version, the models are reduced to a subset of the variables (which we call *probabilistic*) and if the problem is weighted, each model has a weight
and the counter returns the sum of the model's weight.
In addition to that, we impose two more constraints
  - All clauses in the formula F are Horn clauses (of the form `I => h` with `I` a conjuction of literals)
  - The probabilistic variables are partioned into *distributions* such that the weights in each distribution sum up to 1

Schlandals takes as input a file using the following format (subject to change in the short future)
```
c Lines that starts with c are comment lines
c Example of format (modified DIMACS) for a simple bayesian network with three variables A B C with each 2 values
c B is dependent on A and C is depedent on B (this is a chain)
p cnf 16 11
c --- Start of the distributions ---
c Each line define a distribution as well as the index of the probabilistic variables
c a0  a1
d 0.2 0.8
c a0b0 a0b1
d 0.3 0.7
c a1b0 a1b1
d 0.4 0.6
c b0c0 b0c1
d 0.1 0.9
c b1c0 b1c1
d 0.5 0.5
c --- End of the distributions ---
c --- Clauses ---
c Clauses for cpt P(A)
c This clause can be seen as 0 => 10
10 -0
11 -1
c Clauses for cpt P(B | A)
c Multiple negative indexes can be seen as 10 /\ 2 => 12
12 -10 -2
12 -11 -4
13 -10 -3
13 -11 -5
c Clauses for cpt P(C | B)
14 -12 -6
14 -13 -8
15 -12 -7
15 -13 -9
c The query is added by such clauses, which is translated as 11 => False
-11
```

# Usage

To use the solver you must have the Rust toolchain installed. Once this is done you can clone this repository and build the solver with the following commands
```
git clone git@github.com:aia-uclouvain/schlandals.git && cd schlandals && cargo build --release
```
The binary's CLI arguments are organized by commands. We support three commands at the time

### Search

`schlandals search -i <input> -b <branching heuristic> [-s -m <memory limit]` launch the search based solver.
The `i` is a path to a valid input file, `b` is a valid branching heuristic.
The optional `s` flag tells if stats must be recorded or not and `m` can be used to provide a memory limit.

### Compilation

`schlandals compile -i <input> -b <branching heuristic> [--fdac <outfile> --dotfile <dotfile>]` can be used to compile the input problem as an arithmetic circuit.
The circuit can be stored in the file given by the `fdac` argument and a DOT visualization can be saved in the argument provided by the `dotfile` argument.

### Reading a compiled file

A previously compiled file can be read using `schlandals read-compiled -i <fdac file> [--dotfile <dotfile]`.
