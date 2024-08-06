# Modelizing Bayesian Networks

A Bayesian Network (BN) can be seen as a structural way to encode conditional independences of a joint probability distribution.
Let \\( \mathcal{V} = \{V_1, \ldots, V_n\} \\) be random variables, then a Bayesian network is a directed acyclic graph structure over these variables, a show below.

![](bn.svg)

This BN represent a joint probability distribution over binary variables A, B, C, and D.
The directed edges encode the dependences; for example, B and C depends on A. In particular, we have that B and C are independent given C.
Each variable has an associated conditional probability table (CPT) that gives its probability distributions given the values of its parents.
The probabilities are called the parameters of the BN and are denoted by \\( \theta \\).

Currently, Schlandals is able to compute marginal probabilities of the BN (e.g., queries like \\( P(C = \top) \\), \\(P C = \top, B = \bot \\)).
