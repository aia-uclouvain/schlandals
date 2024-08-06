# Schlandals as Compiler

Schlandals can be used as a compiler that output an arithmetic circuit from a CNF formula. In such mode, it first executes the search and then parse the cache to obtain the arithmetic circuit [1].
Hence, the compilation mode should only be used if you actually need the compiled circuit. If you only want to evaluate the probability of a problem, use the search.

## Running the compilation

The compilation can be run using the following command

```bash
schlandals compile -i model.cnf compile
Estimated probability 1.3542085e-2 with bounds [1.3542085e-2 1.3542085e-2] found in 4 seconds
```
This command executes the search, compile the arithmetic circuit from the cache, and evaluates it.
Notice that the compilation accepts the same arguments as the search, since it executes one first. Hence, you can produce approximate arithmetic circuits by doing an approximate search.

## Command Line Arguments
```bash
schlandals compile --help
Usage: schlandals --input <INPUT> compile [OPTIONS]

Options:
      --fdac <FDAC>        If the problem is compiled, store it in this file
      --dotfile <DOTFILE>  If the problem is compiled, store a DOT graphical representation in this file
  -h, --help               Print help
```

**Note**: if you want to use general schlandals option, they must come **before** the `compile` sub-command (e.g. `schlandals -i model.cnf --epsilon 0.2 --approx lds compile --fdac ac.out`)
