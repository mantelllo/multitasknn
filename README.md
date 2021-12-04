Multi-task NN simultaneously solving two related classification and regression problems.

## Problem
Given `X`(dim=6) and `z`(scalar, provided for task 1 only), solve two tasks:
1. classification - `logit(P(y1|X, z)) = B2 * X + b3 * z`,
2. regression - `E(y2|X) = B1 * X`,

with assumption, that `B2 = B1 * b4`.


## Input CSV
Dataset with 10 columns expected:
* `DatasetID` - value "1" for task1, "2" for task2;
* 6 for `X` input values - `x1`, `x2`, ..., `x6`;
* `z` - scalar, only for task1;
* `y1` - binary label for task1;
* `y2` - scalar label for task2.


## Features
This application trains NN with input csv and returns trained `B1`, `B2`, `b3`, `b4` values.
In order to adjust prioritization for task1, run application with flag `--task2-loss-multiplier 0.001` (overfitting warning).
The given dataset may be upsampled/downsampled automatically to balanced dataset.

`Adam` optimizer is used for this task. At first, training is run with `1e-1` learning rate (two epochs), then it is changed to `1e-3`. `0.2` of dataset is used for validation. When training over `val_loss` no longer improves, the training is stopped.
Loss components of both tasks (`y1_loss` and `y2_loss`) will be output separately during training.


## Requirements
Application is built to run on docker, so requirements are only:
* `docker`;
* `git`;
* input csv file (optional, sample file is provided).


## Setup
Download code to your computer using cli commands:
* clone repository with `git clone https://github.com/mantelllo/multitasknn`;
* cd to project directory using `cd multitasknn`;
* add your dataset (csv format) to `data/` directory.


## Run
If custom csv is used, please add it to `./data` directory before building an image.

* Build image with `docker build . -t multitasknn`;
* run container with `docker run multitasknn`;
* for addition flags run `docker run multitasknn -h`.
* to provide custom csv run ` docker run multitasknn --csv-file './data/name_of_other_csv.csv'`


## Sample run

```bash
$ docker run multitasknn
Using ./data/multitasklearnig_task.csv as input file
Epoch 1/2
782/782 [==============================] - 29s 37ms/step - loss: 1.2716 - y1_loss: 0.2070 - y2_loss: 1.0646
Epoch 2/2
782/782 [==============================] - 29s 37ms/step - loss: 1.2253 - y1_loss: 0.1961 - y2_loss: 1.0293
Epoch 1/50
625/625 [==============================] - 28s 44ms/step - loss: 1.2087 - y1_loss: 0.1956 - y2_loss: 1.0131 - val_loss: 1.2157 - val_y1_loss: 0.1956 - val_y2_loss: 1.0201
Epoch 2/50
625/625 [==============================] - 26s 42ms/step - loss: 1.2015 - y1_loss: 0.1839 - y2_loss: 1.0177 - val_loss: 1.2128 - val_y1_loss: 0.1922 - val_y2_loss: 1.0206
Epoch 3/50
625/625 [==============================] - 27s 43ms/step - loss: 1.2010 - y1_loss: 0.1836 - y2_loss: 1.0173 - val_loss: 1.2121 - val_y1_loss: 0.1923 - val_y2_loss: 1.0198
Epoch 4/50
625/625 [==============================] - 27s 43ms/step - loss: 1.2008 - y1_loss: 0.1841 - y2_loss: 1.0167 - val_loss: 1.2114 - val_y1_loss: 0.1922 - val_y2_loss: 1.0192
Epoch 5/50
625/625 [==============================] - 27s 43ms/step - loss: 1.2007 - y1_loss: 0.1844 - y2_loss: 1.0162 - val_loss: 1.2113 - val_y1_loss: 0.1930 - val_y2_loss: 1.0184
Epoch 6/50
625/625 [==============================] - 27s 43ms/step - loss: 1.2042 - y1_loss: 0.1843 - y2_loss: 1.0198 - val_loss: 1.2112 - val_y1_loss: 0.1934 - val_y2_loss: 1.0178
Epoch 7/50
625/625 [==============================] - 27s 43ms/step - loss: 1.2001 - y1_loss: 0.1840 - y2_loss: 1.0161 - val_loss: 1.2114 - val_y1_loss: 0.1942 - val_y2_loss: 1.0172
Epoch 00007: early stopping
Model variables:
B1: [1.822291   2.2256975  2.2108572  2.4122927  0.58907384 1.8887532 ]
B2: [0.65060896 0.79463637 0.78933793 0.86125606 0.21031585 0.6743378 ]
b3: 0.5791565
b4: 0.357028

```
