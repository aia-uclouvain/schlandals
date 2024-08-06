# Modelization

This section explains how the problems must be encoded for the Schlandals solver.
The solver supports multiple file formats, but they rely on the same principle; the probabilistic model (i.e., its structure and parameters) is encoded in a file, and the query is encoded as evidence (i.e., assignment on some of the model's variable) is encoded in a separate file.
For each problem, the evidence can be given as a string in the command line arguments. For example, to compute a marginal probability in a Bayesian network specified in UAI format, one can use `schlandals -i bn.uai --evidence "2 1 0"` where the evidence is UAI-style formatted.

You can find in the following pages a description of the supported file formats for each problem that can be solved using Schlandals. If you would like to use a file format that is not yet supported, consider [opening an issue](https://github.com/aia-uclouvain/schlandals/issues/new). For a more formal description of the Schlandals language, see [1].
Notice that Schlandals works on CNF formulas; hence all the file formats are, in the end, transformed into CNF (the Dimacs file format directly represent the used formulas).

- Bayesian network
    - [Problem description](bn/README.md)
    - [Dimacs-style format](bn/dimacs.md)
    - [UAI format](bn/uai.md)
- Reachability in probabilistic graphs
    - [Problem description](pg/README.md)
    - [Dimacs-style format](pg/dimacs.md)
    - [PG format](pg/pg.md)

[1] Alexandre Dubray, Pierre Schaus, and Siegfried Nijssen. Probabilistic Inference by Projected Weighted Model Counting on Horn Clauses. In 29th International Conference on Principles and Practice of Constraint Programming (CP 2023), 2023
