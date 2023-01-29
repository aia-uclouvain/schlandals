import os
import sys
import re
import itertools
_script_dir = os.path.dirname(os.path.realpath(__file__))

def get_graph(filename):
    with open(os.path.join(_script_dir, filename)) as f:
        nodes = {}
        source = None
        target = None
        first = True
        for line in f:
            if first or line.startswith('d'):
                first = False
                continue
            ls = line.rstrip().split(' ')
            if len(ls) != 3:
                if line.startswith('-'):
                    target = int(ls[0][1:])
                else:
                    source = int(ls[0])
            s = [int(re.sub('-', '', x)) for x in line.rstrip().split(' ')]
            if len(s) != 3:
                continue
            nfrom = s[1]
            nto = s[0]
            try:
                nodes[s[1]].add(s[0])
            except KeyError:
                nodes[s[1]] = {s[0]}
        return nodes, source, target

def find_all_path_(nodes, source, target, visited, paths_cache):
    if source in paths_cache:
        return [{(source, target)}.union(p) for p in paths_cache[source]]
    if source not in nodes:
        return []
    visited.add(source)
    paths = []
    try:
        for n in nodes[source]:
            if n == target:
                paths.append({(source, target)})
                break
            if n not in visited:
                paths += [{(source, n)}.union(p) for p in find_all_path(nodes, n, target)]
    except KeyError:
        pass
    paths_cache[source] = paths
    visited.remove(source)
    return paths

def find_all_path(nodes, source, target):
    return find_all_path_(nodes, source, target, set(), {})

g, source, target = get_graph(sys.argv[1])
paths = find_all_path(g, source, target)
proba = 0.0
for p in paths:
    proba += 0.875**len(p)
substract = True
indexes = [i for i in range(len(paths))]
for k in range(2, len(paths)+1):
    for combination in itertools.combinations(indexes, k):
        edges = set()
        for x in combination:
            edges = edges.union(paths[x])
        if substract:
            proba -= 0.875**len(edges)
        else:
            proba += 0.875**len(edges)
    substract = not substract
print(len(paths), proba)