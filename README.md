![Rust](https://github.com/AlexandreDubray/schlandals/actions/workflows/rust.yml/badge.svg)

Schlandals is a state-of-the-art _Projected Weighted Model Counter_ specialized for probabilistic inference over discrete probability distributions.
Currently, there are known modelization for the following problems
  - Computing the marginal probabilities of a variable in a Bayesian Network
  - Computing the probability that two nodes are connected in a probabilistic graph
  - Computing the probability of [ProbLog](https://github.com/ML-KULeuven/problog) programs

For more information on how to use Schlandals and its mechanics, check [the documentation](https://aia-uclouvain.github.io/schlandals) (still in construction).
You can cite Schlandals using the following bibtex entry
```
@InProceedings{schlandals
  author =	{Dubray, Alexandre and Schaus, Pierre and Nijssen, Siegfried},
  title =	{{Probabilistic Inference by Projected Weighted Model Counting on Horn Clauses}},
  booktitle =	{29th International Conference on Principles and Practice of Constraint Programming (CP 2023)},
  year =	{2023},
  doi =		{10.4230/LIPIcs.CP.2023.15},
}
```

If you use our LDS-based approximation, you can also cite
```
@InProceedings{schlandals_anytime_approximation
  author =	{Dubray, Alexandre and Schaus, Pierre and Nijssen, Siegfried},
  title =	{{Anytime Weighted Model Counting With Approximation Guarantees For Probabilistic Inference}},
  booktitle =	{30th International Conference on Principles and Practice of Constraint Programming (CP 2024)},
  year =	{2024},
}
```
