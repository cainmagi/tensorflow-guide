# 1-3: Non-linear Regression Test

Non-linear regression demo for Tensroflow.
Use non-linear regression to learn the best w, p, a for

y ~ cos( x w^T + 1 p^T ) a,

which could be also representes by

y = sum [ a_i cos( w_i x + p_i ) ]

It contains four files:

* `tools.py`: data post-processing codes.
* `dparser.py`: data processor, which is used to pre-process the data.
* `extension.py`: a module for extending the Tensorflow lib.
* `nonlin-reg.py`: the main module, where we define the network and need to include `dparser`, `extension` as the submodule.

Check the theory and guide on [here](https://cainmagi.github.io/tensorflow-guide/book-1-x/chapter-1/nonlinear-regression/)

Thanks [Radian ticks - matplotlib](https://matplotlib.org/gallery/units/radian_demo.html?highlight=radian%20ticks) for providing an external module to standard matplotlib.

## Usage

Run this command to see the help:

```bash
python nonlin-reg.py -h
```

It supports user defined options.

### Basic Usage

To run the code, we just need to use 

```bash
python nonlin-reg.py
```

In detail, we could use such a command to specify more options:

```bash
python nonlin-reg.py -lr 0.1 -e 30 -se 500 -tbn 32 -tsn 20
```

where we set the learning rate, i.e. `-lr`(`--learningRate`) as 0.1. Then we set the number of epochs `-e`(`--epoch`) as 30. In each epoch, we run 500 steps by specifying `-se`(`--steppe`). In each step, we use 32 samples as a batch (`-tbn`/`--trainBatchNum`). In testing phase, we generate one batch to test the results. This batch would contain `-tsn`(`--testBatchNum`), i.e. 20 samples.

#### Advanced options

1. Use `-do`(`--outputData`) to define path and file name of the saved data. (do not need to add postfix)。 Here is the command:

    ```bash
    python nonlin-reg.py -do test/algorithm/base
    ```
    
    Would create `./test/algorithm/base.npz`.
    
2. Use `-o`(`--optimizer`) to specify the name of the optimizer, we support `adam`, `amsgrad`, `adamax`, `nadam`(Nesterov Adam), `adadelta`, `rms`(RMSprop), `adagrad`, `nmoment` (Nesterov momentum), 'sgd' (stochastic gradient descent). Here is the command:
    
    ```bash
    python nonlin-reg.py -o amsgrad
    ```
    
    The option is caseless.
    
3. Use `-xl`(`--xLength`) to define the length of the random vector which is used as the input:

    ```bash
    python nonlin-reg.py -xl 1000
    ```
    
4. Use `-sd`(`--seed`) to set the random seed for the experiment, this option would make the results reproductable. Here is the command:

    ```bash
    python nonlin-reg.py -sd 1
    ```

# Update records

## 1.0 @ 03/17/2019

Finish this project.

## 0.5 @ 03/15/2019

Create this project.