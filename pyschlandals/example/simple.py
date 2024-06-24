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

p2 = problem.copy()

problem.add_clause([-15])
p2.add_clause([-16])
print(problem.solve())
print(p2.solve())
