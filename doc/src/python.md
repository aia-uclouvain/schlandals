A Python interface, `pyschlandals`, is available [from PyPi here](https://pypi.org/project/pyschlandals/).
The interface is still rudimentary; open a pull request if you need any functionality.
To install the Python interface, run `pip install pyschlandals`.

## Running a simple problem

A problem in `pyschlandals` is a set of distributions and clauses.
The following code block shows how to create a simple problem instance for a Bayesian Network and solve it using the DPLL-based search.
Notice that the indexes in the clauses start at 1, and the distributions use the first indexes.

```python
from pyschlandals.pwmc import PyProblem

problem = PyProblem()
problem.add_distribution([0.2, 0.8])
problem.add_distribution([0.3, 0.7])
problem.add_distribution([0.4, 0.6])
problem.add_distribution([0.1, 0.9])
problem.add_distribution([0.5, 0.5])

problem.add_clause([11, -1])
problem.add_clause([12, -2])
problem.add_clause([13, -11, -3])
problem.add_clause([13, -12, -5])
problem.add_clause([14, -11, -4])
problem.add_clause([14, -12, -6])
problem.add_clause([15, -13, -7])
problem.add_clause([15, -14, -9])
problem.add_clause([16, -13, -8])
problem.add_clause([16, -14, -10])
problem.add_clause([-15])

print(problem.solve())
```

The problem generation can be seen as lazy. The `PyProblem` is sent to the rust code only when `problem.solve()` is called.
At this point, the distributions and clauses are sent to Schlandals as an alternative to the `.cnf` files.
