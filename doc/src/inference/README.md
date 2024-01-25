# Probabilistic Inference

Schlandals provides exact and \\( \epsilon \\)-bounded approximate inference methods.
It can be used either as a DPLL-style search-based solver or as a compiler that outputs an arithmetic circuit.
Beware that, currently; the compiler is just a search-solver saving its trace. Hence, you should use the search solver if you do not need to keep the circuits for further computation.
