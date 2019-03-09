# 1-1: Linear Classification Test

Linear classification demo for Tensroflow.
Use logistic regression to learn the best W,b for

y ~ W x + b.

It contains two files:

* `dparser.py`: data processor, which is used to pre-process the data.
* `lin-cls.py`: the main module, where we define the network and need to include `dparser` as the submodule.

Check the theory and guide on [here](https://cainmagi.github.io/tensorflow-guide/book-1-x/chapter-1/linear-classification/)

Run this command to see the performance:

```bash
python lin-cls.py
```

# Update records

## 1.1 @ 03/09/2019

Upgrade the API version from r1.12 to r1.13.The modifications are mainly about definitions of the losses and metrics.

## 1.0 @ 03/05/2019

Create this project.