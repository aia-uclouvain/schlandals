from pyschlandals.search import exact
from pyschlandals.compiler import compile
from pyschlandals import BranchingHeuristic

filename = '../tests/instances/bayesian_networks/asia_xray_false.cnf'
print(exact(filename, BranchingHeuristic.MinInDegree))

dac = compile(filename, BranchingHeuristic.MinInDegree)
print(dac.get_circuit_probability())