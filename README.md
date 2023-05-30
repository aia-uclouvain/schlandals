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
Then you can run the solver with `./target/release/schlandals [OPTIONS] --input <input file> --branching <branching heuristic>`.

The available branching heuristics all selects a clause in the input file and then a distribution randomly from it. They work on the *implication graph* of the clauses (one node per clause, an edge from a clause `I => j` to another clause `K => l` if `j` is included in `K`). We currently provide the following heuristics
  - `fiedler`: computes the fiedler vector of the implication graph and selects the clause with the value closest to the mean fiedler value of all active clauses
  - `min-in-degree`: selects the active clause with the less number of parents. In case of ties selects the one with the fewest number of initial parents.
  - `min-out-degree`: selects the active clause with the less number of children. In case of ties selects the one with the fewest number of initial children
  - `max-degree`: selects the clause with the maximum degree.

The available options are
  - `compiled`: Compile the input into an AOMDD. If active, the following options are available:
      - `faomdd`: The file from which to read the AOMDD. If present, the AOMDD is constructed from the file and evaluated.
      - `fgraphviz`: Save the AOMDD as a graphviz DOT file for visualization
  - `memory`: Limit on the memory usage for the search-based solver (not active when compiling into an AOMDD)
  - `statistics`: Shows statistics of the search
