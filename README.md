Multi-task NN simultaneously solving two related classification and regression problems.

## Requirements
* docker;
* input csv file (sample file is given).

## Setup
* `git clone https://github.com/mantelllo/multitasknn`;
* `cd multitasknn`;
* add dataset in csv format to `data/` directory.

## RUN
* build image with `docker build . -t multitasknn`;
* run container with `docker run multitasknn`;
* for addition flags run `docker run multitasknn -h`.
