# The Language of Schlandals

This page gives a detailed explanation of the language used by Schlandals. It also describes a modified DIMACS format that the solver uses. We assume basic knowledge of propositional logic.

## A Restricted Propositional Logic Language

Schlandals works on propositional formulas in Conjunctive Normal Form (CNF) containing only Horn clauses.
In Horn clauses, at most one literal has a positive polarity.
For example, \\( \lnot x \lor \lnot y \lor z \\) is a valid Horn clause as only \\( z \\) has a positive polarity.
Using the logical equivalence \\( (A \implies B) \Leftrightarrow (\lnot A \lor B) \\), Horn clauses are implications with potentially empty consequences.

Let \\( F \\) be a Horn formula over variables \\( X \\). Then, in Schlandals, the variables are partitioned into two sets \\( P \\) and \\( D \\).
The set \\( D \\) of variables contains classical boolean variables (deterministic variables), while the set \\(P\\) includes the variables associated with the distributions (probabilistic variables).
That is, \\( P = \cup_{i = 1}^n D_i \\) such that \\( D_i \cap D_j = \emptyset \\) for all distinct distributions. As usual, we have \\( \sum_{v \in D_i} P(D_i = v) = 1 \\)
For an interpretation \\(I\\) of \\(F\\) to be a model (i.e., \\(F[I] = T\\)), it must set **exactly one variable to T in each distribution**.

## A Modified DIMACS Format

Schlandals reads its inputs from a modified [DIMACS format](https://mccompetition.org/assets/files/2021/competition2021.pdf).
Basically, it follows the same rules as the DIMACS format, with an additional rule for the distributions (and it ignores the lines starting with `c p weight` which are used in other weighted model counters).
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
