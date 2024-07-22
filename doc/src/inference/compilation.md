# Schlandals as Compiler

Schlandals can be used as a compiler that output an arithmetic circuit from a CNF formula. In such mode, it first executes the search and then parse the cache to obtain the arithmetic circuit [1].
Hence, the compilation mode should only be used if you actually need the compiled circuit. If you only want to evaluate the probability of a problem, use the search.

## Running the compilation

The compilation can be run using the following command

```bash
schlandals compile -i model.cnf
Estimated probability 1.3542085e-2 with bounds [1.3542085e-2 1.3542085e-2] found in 4 seconds
```
This command executes the search, compile the arithmetic circuit from the cache, and evaluates it.
Notice that the compilation accepts the same arguments as the search, since it executes one first. Hence, you can produce approximate arithmetic circuits by doing an approximate search.

## Command Line Arguments
```bash
schlandals compile --help
Use the DPLL-search structure to produce an arithmetic circuit for the problem

Usage: schlandals compile [OPTIONS] --input <INPUT>

Options:
  -i, --input <INPUT>
          The input file

  -b, --branching <BRANCHING>
          How to branch

          [default: min-in-degree]

          Possible values:
          - min-in-degree: Minimum In-degree of a clause in the implication-graph

      --fdac <FDAC>
          If present, store a textual representation of the compiled circuit

      --dotfile <DOTFILE>
          If present, store a DOT representation of the compiled circuit

  -e, --epsilon <EPSILON>
          Epsilon, the quality of the approximation (must be between greater or equal to 0). If 0 or absent, performs exact search

  -a, --approx <APPROX>
          If epsilon present, use the appropriate approximate method

          [default: bounds]

          Possible values:
          - bounds: Bound-based pruning
          - lds:    Limited Discrepancy Search

  -h, --help
          Print help (see a summary with '-h')
```
