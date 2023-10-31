![Rust](https://github.com/AlexandreDubray/schlandals/actions/workflows/rust.yml/badge.svg)
[![codecov](https://codecov.io/gh/AlexandreDubray/schlandals/branch/main/graph/badge.svg?token=J4J2I9Q9KX)](https://codecov.io/gh/AlexandreDubray/schlandals)

**The code is still in early development and should not be considered stable**

# schlandals

Schlandals is a projected weighted model counter which targets inference in probabilistic models.
Current known probability queries that can be solved with the solver include
  - Computing the probability of some target variables (given some evidences) in a Bayesian network
  - Computing the probability that two nodes are connected in a probabilistic graph
  - A mapping from [ProbLog](https://github.com/ML-KULeuven/problog) programs to Schlandals

# Problem specification

A model counter is a  program that counts the number of satisfying assignments of a boolean formula F.
In its projected version, the models are reduced to a subset of the variables (which we call *probabilistic*) and if the problem is weighted, each model has a weight
and the counter returns the sum of the model's weight.
In addition to that, we impose two more constraints
  - All clauses in the formula F are Horn clauses (of the form `I => h` with `I` a conjuction of literals)
  - The probabilistic variables are partioned into *distributions* such that the weights in each distribution sum up to 1

Schlandals takes as input a file using a modified version of the [DIMACS](https://mccompetition.org/assets/files/2021/competition2021.pdf) format
```
c Lines starting with `c` alone are comments.
c The first line must be the head in the form of "p cnf <number variable> <number clauses>
p cnf 16 11
c The distribution are defined by lines starting with "c p distribution"
c Each value in a distribution is assigned to a variable, starting from 1. Hence, the first N variables
c are used for the distributions.
c p distribution 0.2 0.8
c p distribution 0.3 0.7
c p distribution 0.4 0.6
c p distribution 0.1 0.9
c p distribution 0.5 0.5
c Finally the clauses are encoded as in DIMACS.
11 -1 0
12 -2 0
13 -11 -3 0
13 -12 -5 0
14 -11 -4 0
14 -12 -6 0
15 -13 -7 0
15 -14 -9 0
16 -13 -8 0
16 -14 -10 0
-12 0
```

# Usage

To use the solver you must have the Rust toolchain installed. Once this is done you can clone this repository and build the solver with the following commands
```
git clone git@github.com:aia-uclouvain/schlandals.git && cd schlandals && cargo build --release
```
Once the command has been built, the binary is located in `<SCHLANDALS_DIR>/target/release/schlandals` (with `<SCHLANDALS_DIR>` the directory in which you cloned the repo).
If you want to be able to run Schlandals for anywhere, you can add `<SCHLANDALS_DIR>/target/release` to your `$PATH`.

The binary's CLI arguments are organized by commands. Three commands are currently supported: `search` to solve the problem by a DPLL search, `compile` to compile the input into an arithmetic circuit and `read-compiled` to evaluate a previously compiled input.
These are explained next, after a quick note on the heuristics supported by the solver (for the `search` and `compile` sub-command)
### Heuristics

Schlandals comes with various heuristic that can be used during the search/compilation.
The current heuristics are based on the implication graph of the input. In such graph, there is one node per clause and a link between clause C1 (I1 => h1) and C2 (I2 => h2) if the h1 is in I2.
The available heuristics are
  - `min-in-degree`: Select a distribution from the clause with the lowest in degree.
  - `min-out-degree`: Select a distribution from the clause with the lowest out degree.
  - `max-degree`: Select a distribution from the clause with the highest degree.

### Search

`schlandals search -i <input> -b <branching heuristic> [-s -m <memory limit]` launch the search based solver.
The `i` is a path to a valid input file, `b` is a valid branching heuristic.
The optional `s` flag tells if stats must be recorded or not and `m` can be used to provide a memory limit.

### Compilation

`schlandals compile -i <input> -b <branching heuristic> [--fdac <outfile> --dotfile <dotfile>]` can be used to compile the input problem as an arithmetic circuit.
The circuit can be stored in the file given by the `fdac` argument and a DOT visualization can be saved in the argument provided by the `dotfile` argument.

### Reading a compiled file

A previously compiled file can be read using `schlandals read-compiled -i <fdac file> [--dotfile <dotfile]`.

# Citing

To cite Schlandals in your work, use the following

```
@InProceedings{dubray_et_al:LIPIcs.CP.2023.15,
  author =	{Dubray, Alexandre and Schaus, Pierre and Nijssen, Siegfried},
  title =	{{Probabilistic Inference by Projected Weighted Model Counting on Horn Clauses}},
  booktitle =	{29th International Conference on Principles and Practice of Constraint Programming (CP 2023)},
  pages =	{15:1--15:17},
  series =	{Leibniz International Proceedings in Informatics (LIPIcs)},
  ISBN =	{978-3-95977-300-3},
  ISSN =	{1868-8969},
  year =	{2023},
  volume =	{280},
  editor =	{Yap, Roland H. C.},
  publisher =	{Schloss Dagstuhl -- Leibniz-Zentrum f{\"u}r Informatik},
  address =	{Dagstuhl, Germany},
  URL =		{https://drops.dagstuhl.de/opus/volltexte/2023/19052},
  URN =		{urn:nbn:de:0030-drops-190520},
  doi =		{10.4230/LIPIcs.CP.2023.15},
  annote =	{Keywords: Model Counting, Bayesian Networks, Probabilistic Networks}
}
```