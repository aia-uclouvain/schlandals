# Modelization

In this section, we describe how to encode problems in Schlandals.
We first give the theoretical foundations to understand these encodings, and then each sub-section shows how to encode a specific problem.
We currently have encodings (or translations) for

- [Bayesian Networks](./bn.md)
- [Reachability in Probabilistic Graphs](./pg.md)
- [ProbLog Programs](./problog.md)

## A Restricted Propositional Logic Language

Schlandals works on propositional formulas in Conjunctive Normal Form (CNF) containing only Horn clauses.
In Horn clauses, at most one literal has a positive polarity.
For example, \\( \lnot x \lor \lnot y \lor z \\) is a valid Horn clause as only \\( z \\) has a positive polarity.
Using the logical equivalence \\( (A \implies B) \Leftrightarrow (\lnot A \lor B) \\), Horn clauses are implications with potentially empty consequences.

Let \\( F \\) be a Horn formula over variables \\( \mathcal{X} \\). In Schlandals, the variables are partitioned into two sets \\( \mathcal{P} \\) and \\( \mathcal{Y} \\).
The set \\( \mathcal{Y} \\) of variables contains classical boolean variables (deterministic variables), while the set \\( \mathcal{P} \\) includes the variables associated with the distributions (probabilistic variables).
Let us assume that \\( n \\) distributions are defined for the problem.
Then, \\( \mathcal{P} = \cup_{i = 1}^n D_i \\) with \\( D_i \cap D_j = \emptyset \\) for all \\( i \neq j \\).
Moreove, every variable \\( v \in D \\) has a weight \\( P(v) \\) and \\( \sum_{v \in D_i} P(v) = 1 \\).

Let \\( I = \mathbf{x} \\) be an assignment to the variables \\( \mathcal{X} \\). We say that \\( I \\) is an interpretation of \\( F \\), and it is a model if \\( F[I] = \top \\) (i.e., evaluating \\( F \\) with the values in \\( I \\) reduce to true).
In Schlandals, there is an additional constraint: a model \\( I \\) is valid only if it sets exactly one variable to \\( \top \\) per distribution.

If we denote \\( v_i \\) the value set to \\( \top \\) in distribution \\( D_i \\), the weight of a model \\( I \\) can be computed as \\[ \omega(I) = \prod_{i = 1}^n P(v_i) \\]
If \\( \mathcal{M} \\) denotes the set of models of \\( F \\), then the goal is to compute \\[ \sum_{I \in \mathcal{M}} \omega(I) \\].

## A Modified DIMACS Format

Schlandals reads its inputs from a modified [DIMACS format](https://mccompetition.org/assets/files/2021/competition2021.pdf).
It follows the same rules as the DIMACS format, with an additional rule for the distributions (and it ignores the lines starting with `c p weight` used in other weighted model counters).
Below is an example of a modified DIMACS file
```
p cnf 16 11
c Lines starting with `c` alone are comments.
c The first line must be the head in the form of "p cnf <number variable> <number clauses>
c The distributions are defined by lines starting with "c p distribution"
c Each value in a distribution is assigned to a variable, starting from 1. Hence, the first N variables
c are used for the distributions.
c p distribution 0.2 0.8
c p distribution 0.3 0.7
c p distribution 0.4 0.6
c p distribution 0.1 0.9
c p distribution 0.5 0.5
c Finally, the clauses are encoded as in DIMACS.
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
