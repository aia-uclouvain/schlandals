# Search-based Inference

When solving an inference problem, Schlandals always performs a DPLL-style search over the solution space (even when compiling into an arithmetic circuit).
However, due to the modelization choices made in Schlandals, there are few differences with classical DPLL search:
1. The branching is done on the distributions, and not the variables. Hence, the "variable selection" heuristic is replaced with a "distribution selection" heuristic
2. When there are no more distribution to branch on, if the Boolean Unit Propagation (BUP) did not return a failure, the remaining problem is SAT. This is due to the fact that there are only Horn clauses in the problem and the SAT problem for Horn formula can be solved with BUP.
3. On top of that, there is an additional propagation, specific to Schlandals, relying on the Horn structure of the problem. For a detailed description of this propagation, see [1].

## Exact inference

An exact inference strategy can be launched with the following command

```bash
[schlandals@schlandalspc]$ schlandals search -i model.cnf
Estimated probability 1.1029004e-1 with bounds [1.1029004e-1 1.1029004e-1] found in 0 seconds
```

The solver output bounds on the probability and the time needed to solve the problem. For larger problem, it is possible to add a timeout and the bounds might not be tight. For a complete description on how the bounds are computed, see [2].
```bash
[schlandals@schlandalspc]$ schlandals --timeout 30 search -i large_model.cnf
Estimated probability 0.353553 with bounds [0.25 0.5] found in 30 seconds
```

## Approximate inference

Schlandals can also perform anytime approximate inference with \\( \varepsilon \\)-guarantees [2]. If \\( p \\) is the true probability, it returns a probability \\( \tilde p \\) such that \\[ \frac{p}{1 + \varepsilon} \leq \tilde p \leq p (1 + \varepsilon) \\].
To use such approximate algorithm, add the `--epsilon` command line argument
```bash
[schlandals@schlandalspc]$ schlandals search -i model.cnf --epsilon 0.01 --approx lds
Estimated probability 0 with bounds [0 5.0653832e-1] found in 0 seconds
Estimated probability 0 with bounds [0 2.7010540e-1] found in 0 seconds
Estimated probability 4.7705749e-3 with bounds [1.7279254e-4 1.3170931e-1] found in 0 seconds
Estimated probability 1.5684742e-2 with bounds [2.9445549e-3 8.3547817e-2] found in 0 seconds
Estimated probability 1.8642883e-2 with bounds [7.2711825e-3 4.7799252e-2] found in 0 seconds
Estimated probability 1.6748061e-2 with bounds [1.0781966e-2 2.6015435e-2] found in 0 seconds
Estimated probability 1.5110826e-2 with bounds [1.2312104e-2 1.8545738e-2] found in 0 seconds
Estimated probability 1.4143238e-2 with bounds [1.3100930e-2 1.5268473e-2] found in 1 seconds
Estimated probability 1.3766330e-2 with bounds [1.3381450e-2 1.4162279e-2] found in 2 seconds
Estimated probability 1.3616029e-2 with bounds [1.3488540e-2 1.3744723e-2] found in 4 seconds
Estimated probability 1.3616029e-2 with bounds [1.3488540e-2 1.3744723e-2] found in 4 seconds
```

Notice that you can also do exact anytime inference with
```bash
[schlandals@schlandalspc]$ schlandals search -i model.cnf --approx lds
Estimated probability 0 with bounds [0 5.0653832e-1] found in 0 seconds
Estimated probability 0 with bounds [0 2.7010540e-1] found in 0 seconds
Estimated probability 4.7705749e-3 with bounds [1.7279254e-4 1.3170931e-1] found in 0 seconds
Estimated probability 1.5684742e-2 with bounds [2.9445549e-3 8.3547817e-2] found in 0 seconds
Estimated probability 1.8642883e-2 with bounds [7.2711825e-3 4.7799252e-2] found in 0 seconds
Estimated probability 1.6748061e-2 with bounds [1.0781966e-2 2.6015435e-2] found in 0 seconds
Estimated probability 1.5110826e-2 with bounds [1.2312104e-2 1.8545738e-2] found in 1 seconds
Estimated probability 1.4143238e-2 with bounds [1.3100930e-2 1.5268473e-2] found in 1 seconds
Estimated probability 1.3766330e-2 with bounds [1.3381450e-2 1.4162279e-2] found in 2 seconds
Estimated probability 1.3616029e-2 with bounds [1.3488540e-2 1.3744723e-2] found in 4 seconds
Estimated probability 1.3565927e-2 with bounds [1.3524913e-2 1.3607065e-2] found in 6 seconds
Estimated probability 1.3548943e-2 with bounds [1.3537128e-2 1.3560769e-2] found in 8 seconds
Estimated probability 1.3543849e-2 with bounds [1.3540853e-2 1.3546845e-2] found in 10 seconds
Estimated probability 1.3542610e-2 with bounds [1.3541749e-2 1.3543471e-2] found in 11 seconds
Estimated probability 1.3542321e-2 with bounds [1.3541960e-2 1.3542683e-2] found in 13 seconds
Estimated probability 1.3542321e-2 with bounds [1.3541960e-2 1.3542683e-2] found in 13 seconds
```

## Command line options

```bash
[schlandals@schlandalspc]$ schlandals search --help
DPLL-style search based solver

Usage: schlandals search [OPTIONS] --input <INPUT>

Options:
  -i, --input <INPUT>
          The input file

  -b, --branching <BRANCHING>
          How to branch

          [default: min-in-degree]

          Possible values:
          - min-in-degree: Minimum In-degree of a clause in the implication-graph

  -s, --statistics
          Collect stats during the search, default yes

  -m, --memory <MEMORY>
          The memory limit, in mega-bytes

  -e, --epsilon <EPSILON>
          Epsilon, the quality of the approximation (must be between greater or equal to 0). If 0 or absent, performs exact search

          [default: 0]

  -a, --approx <APPROX>
          If epsilon present, use the appropriate approximate method

          [default: bounds]

          Possible values:
          - bounds: Bound-based pruning
          - lds:    Limited Discrepancy Search

  -h, --help
          Print help (see a summary with '-h')
```

## References

[1] Alexandre Dubray, Pierre Schaus, and Siegfried Nijssen. Probabilistic Inference by Projected Weighted Model Counting on Horn Clauses. In 29th International Conference on Principles and Practice of Constraint Programming (CP 2023), 2023

[2] Alexandre Dubray, Pierre Schaus, and Siegfried Nijssen. Anytime Weighted Model Counting with Approximation Guarantees For Probabilistic Inference. In 30th International Conference on Principles and Practice of Constraint Programming (CP 2024), 2024
