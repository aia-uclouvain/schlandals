# PG file format

The PG file format is a simple file format for encoding the structure of a probabilistic graph.
When using such format, the evidence (source and target nodes) must be given either in a separate file or as a string in the command line arguments.

### File structure

The first line of the file is either `DIRECTED` or `UNDIRECTED` and indicates the type of graph.
Then, the edges and their probability of being present are listed as 3-tuples (the two nodes and the probability).
For example, the following graph

![](pg.svg)

is encoded as follows

```
DIRECTED
A B 0.4
A C 0.8
B D 0.5
C D 0.6
C E 0.7
D E 0.3
```

Then, the evidences can be given as follows
```
A D
```
which ask to compute the probability that A and D are disconnected.
