from pyschlandals import BranchingHeuristic
from pyschlandals.compiler import compile

import sys

if __name__ == '__main__':
    dac = compile(sys.argv[1], BranchingHeuristic.MinInDegree)
    print(dac.get_circuit_probability())