# Learning Settings

## Basic Setting

To initiate training, a minimum requirement is an input file in CSV format. This file should include a list of CNF files, describing the queries to learn from, along with the expected output for each query. The CSV structure should follow the one of following example (saved as train.csv). 
```
File, Probability
path_query1.cnf, 0.6
path_query2.cnf, 0.2
path_query3.cnf, 0.8
```
Queries within the same CSV file should pertain to the same probabilistic problem and thus share a common set of distributions.

The initial distribution values for training are derived from the first CNF file in the query list. Users can choose these initial values e.g., randomly or based on prior knowledge of the probabilistic problem.

The learning can be initiated with the default parameters using the following command.
```
schlandals learn --trainfile train.csv
```

In this basic setup, arithmetic circuits based on queries are compiled exactly.

## Partial Compilation

Schlandals supports learning distribution parameters for partially compiled arithmetic circuits, enabling the handling of queries that may be too large for full compilation.
To use such partial compilation with the learning, the CNF files containing the query description should include an additional line specifying the distributions to be part of the compiled circuit and subject to learning.
The line is of the form `c p learn`, followed by the index of the distributions to be learned.

For instance, if a query involves 5 distributions in its inputs and one wish to use a partially compiled circuit to learn distributions 1, 4, and 5 only, the corresponding line to include in the CNF file would be:
```
c p learn 1 4 5
```

In this partial compilation scenario, the values of the non-learned distributions (e.g., distributions 2 and 3 in our example) will be the values provided in the CNF file. During the learning process, these values of non-learned distributions will be considered as their true values while learning the parameters of the specified distributions.

The command to perform partial compilation is the following. (Assuming the CSV file referring to CNF files with partial compilation indications is saved as `train_partial.csv`).
```
schlandals learn --trainfile train_partial.csv
```

By default, this command performs learning on partially compiled queries using an epsilon value of 0. To set epsilon to a non-null value (e.g., 0.3), use the following command:
```
schlandals learn --trainfile train_partial.csv --epsilon 0.3
```

## Train-Test Setting

The learning module supports the use of a test set, allowing the assessment of learned distribution parameters on unseen queries related to the same probabilistic problem and sharing the same distributions.

A test CSV file containing the list of the test queries can be provided to the learning. The structure of the test file is the same to the one used for the train CSV file.

The test queries will be compiled partially or not, depending on the CNF file. Then, the queries will be evaluated a first time with the initial values of the distributions, and a second time with the learned values of the distributions.

The command to perform the learning with a test set is the following. (Assuming the CSV file we want to learn from is saved as `train.csv` and the CSV file with the test queries is saved as `test.csv`).
```
schlandals learn --trainfile train.csv --testfile test.csv
```

## Logging Training Epochs (and Test Results)

To log losses and predicted outputs of each query at each epoch of learning, use the `--do-log` option along with the `--outputdir` option, indicating the folder for log storage. If a test set is provided, a second CSV file is created with losses and predicted outputs of each test query before and after learning.

The command to perform the learning with logging is the following. (Assuming the folder in which we want the output files to be saved is `logs/`).
```
schlandals learn --trainfile train.csv --testfile test.csv --do-log --outputdir logs/
```

## Command Line Options

```
[schlandals@schlandalspc]$ schlandals learn --help
Learn distribution parameters from a set of queries

Usage: schlandals learn [OPTIONS] --trainfile <TRAINFILE>

Options:
      --trainfile <TRAINFILE>
          The csv file containing the cnf filenames and the associated expected output

      --testfile <TESTFILE>
          The csv file containing the test cnf filenames and the associated expected output

  -b, --branching <BRANCHING>
          How to branch
          
          [default: min-in-degree]

          Possible values:
          - min-in-degree:  Minimum In-degree of a clause in the implication-graph
          - min-out-degree: Minimum Out-degree of a clause in the implication-graph
          - max-degree:     Maximum degree of a clause in the implication-graph
          - vsids:          Variable State Independent Decaying Sum

      --outfolder <OUTFOLDER>
          If present, folder in which to store the output files

  -l, --lr <LR>
          Learning rate
          
          [default: 0.3]

      --nepochs <NEPOCHS>
          Number of epochs
          
          [default: 6000]

  -d, --do-log
          If present, save a detailled csv of the training and use a codified output filename

      --timeout <TIMEOUT>
          If present, define the learning timeout
          
          [default: 18446744073709551615]

  -e, --epsilon <EPSILON>
          If present, the epsilon used for the approximation. Value set by default to 0, thus performing exact search
          
          [default: 0]

      --loss <LOSS>
          Loss to use for the training, default is the MAE Possible values: MAE, MSE
          
          [default: mae]
          [possible values: mae, mse]

  -j, --jobs <JOBS>
          Number of threads to use for the evaluation of the DACs
          
          [default: 1]

  -s, --semiring <SEMIRING>
          The semiring on which to evaluate the circuits. If `tensor`, use torch to compute the gradients. If `probability`, use custom efficient backpropagations
          
          [default: probability]
          [possible values: probability, tensor]

  -o, --optimizer <OPTIMIZER>
          The optimizer to use if `tensor` is selected as semiring
          
          [default: sgd]
          [possible values: adam, sgd]

      --lr-drop <LR_DROP>
          The drop in the learning rate to apply at each step
          
          [default: 0.75]

      --epoch-drop <EPOCH_DROP>
          The number of epochs after which to drop the learning rate (i.e. the learning rate is multiplied by `lr_drop`)
          
          [default: 100]

      --stopping-criterion <STOPPING_CRITERION>
          The stopping criterion for the training (i.e. if the loss is below this value, stop the training)
          
          [default: 0.0001]

      --delta-early-stop <DELTA_EARLY_STOP>
          The minimum of improvement in the loss to consider that the training is still improving (i.e. if the loss is below this value for a number of epochs, stop the training)
          
          [default: 0.00001]

      --patience <PATIENCE>
          The number of epochs to wait before stopping the training if the loss is not improving (i.e. if the loss is below this value for a number of epochs, stop the training)
          
          [default: 5]

  -h, --help
          Print help (see a summary with '-h')
```