# Modelizing Probabilistic Graph

Let \\( G = (V, E) \\) be a graph (undirected or directed) such that each edge \\( e \in E \\) is present in the graph with a probability \\( p(e) \\).
Given a source \\( s \\) and a target \\( t \\) the goal is to compute the probability that \\( s \\) and \\( t \\) are not connected (Note that this 
is the complementary probability that they are connected).
Below is an example of such graph, with five nodes and a query would be to compute the probability that \\( A \\) and \\( E \\) are connected.

![](pg.svg)
