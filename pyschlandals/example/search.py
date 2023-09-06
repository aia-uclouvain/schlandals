from pyschlandals import BranchingHeuristic
from pyschlandals.search import exact

import sys

if __name__ == '__main__':
    proba = exact(sys.argv[1], BranchingHeuristic.MinInDegree)
    print(proba)