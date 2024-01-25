# Search-based Inference

The search-based inference algorithm is a modified DPLL search that branches over the distributions and has specialized propagation for Horn clauses. For a complete description of the algorithm, see [1].

As an example, let use the Bayesian Network from [the encoding page](../modelization/bn.md), which encoding is shown below (assuming it is saved in a file name `bn.cnf`)
```
p cnf 26 19
c p distribution 0.2 0.8
c p distribution 0.6 0.4
c p distribution 0.3 0.7
c p distribution 0.25 0.75
c p distribution 0.75 0.25
c p distribution 1.0 0.0
c p distribution 0.35 0.65
c p distribution 0.8 0.2
c p distribution 0.0 1.0
-1 19 0
-2 20 0
-3 -19 21 0
-4 -19 22 0
-5 -20 21 0
-6 -20 22 0
-7 -19 23 0
-8 -19 24 0
-9 -20 23 0
-10 -20 24 0
-11 -21 -23 25 0
-12 -21 -23 26 0
-13 -21 -24 25 0
-14 -21 -24 26 0
-15 -22 -23 25 0
-16 -22 -23 26 0
-17 -22 -24 25 0
-18 -22 -24 26 0
-26 0
```


## Exact search

The easiest way to run the solver is by using the following command, which outputs a probability of 0.6145.
```
schlandals search -i bn.cnf
```

If you want to run with a limit on the memory used by Schlandals, use
```
schlandals search -i bn.cnf -m 1000
```
This command runs the solver with a memory limit of 1GB. Currently, no strategy is implemented to clean the cache smartly; when the memory limit is reached, the cache is completely cleared, and the exploration continues.

## Approximate inference
Schlandals can provide approximate probability with \\( \epsilon \\) error bounds. That is, if \\( p \\) is the true probability, an \\( \epsilon \\)-bounded approximation \\( \tilde p \\) is such that \\[ \frac{p}{1 + \epsilon} \leq \tilde p \leq p (1 + \epsilon) \\].
To use such approximate algorithm, add the `--epsilon` command line argument
```
schlandals search -i bn.cnf -e 0.3
```
Notice that running the solver with `-e 0.0` performs an exact search.

## Command line options

```
[schlandals@schlandalspc] schlandals search --help
DPLL-style search based solver

Usage: schlandals search [OPTIONS] --input <INPUT>

Options:
  -i, --input <INPUT>
          The input file

  -b, --branching <BRANCHING>
          How to branch
          
          [default: min-in-degree]

          Possible values:
          - min-in-degree:  Minimum In-degree of a clause in the implication-graph
          - min-out-degree: Minimum Out-degree of a clause in the implication-graph
          - max-degree:     Maximum degree of a clause in the implication-graph
          - vsids:          Variable State Independent Decaying Sum

  -s, --statistics
          Collect stats during the search, default yes

  -m, --memory <MEMORY>
          The memory limit, in mega-bytes

  -e, --epsilon <EPSILON>
          Epsilon, the quality of the approximation (must be between greater or equal to 0). If 0 or absent, performs exact search

  -h, --help
          Print help (see a summary with '-h')
```

## References

[1] Alexandre Dubray, Pierre Schaus, and Siegfried Nijssen. Probabilistic Inference by Projected Weighted Model Counting on Horn Clauses. In 29th International Conference on Principles and Practice of Constraint Programming (CP 2023), 2023
