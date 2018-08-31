# Cosmic-ray mass composition analysis

[![Build Status](https://travis-ci.org/jrbourbeau/cr-composition.svg?branch=master)](https://travis-ci.org/jrbourbeau/cr-composition)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
[![GitHub license](https://img.shields.io/github/license/jrbourbeau/cr-composition.svg)](https://github.com/jrbourbeau/cr-composition/blob/master/LICENSE)


*NOTE*: This analysis is still under active development

This repository contains the analysis code used to study the cosmic-ray
composition spectrum using data collected by the IceCube South Pole Neutrino
Observatory.


## Repository layout

The layout of this repository is:

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make processing` or `make plots`
├── README.md          <- Top-level README for developers using this project.
├── config.yml         <- Filesystem path configuration file
│
├── models             <- Scripts for fitting and saving models
│
├── notebooks          <- Jupyter notebooks used for exploratory analysis
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── comptools          <- Python package containing analysis-related code
│
├── processing         <- Scripts for processing simulation and data
│
└── plotting           <- Scripts for making plots
```

While not generated with, this layout for this project was inspired by the
[Cookiecutter Data Science project](https://github.com/drivendata/cookiecutter-data-science).


## Installation

See the [installation and setup](https://jrbourbeau.github.io/cr-composition/setup.html)
section of the documentation.

## Analysis steps

The workflow for this analysis is managed using GNU `make`. Various commands
are stored in `Makefile` and can be executed using the syntax
`make <command>`. The available commands in `Makefile` are shown below

| Name        | Command           | Description  |
|:-------------:|:-------------:| :-----|
| `processing-cluster`      | `make processing-cluster` | Submits data and simulation processing cluster jobs |
| `processing`      | `make processing` | Runs all other necessary processing (e.g. calculates efficiencies, fits ML models, etc.) |
| `models`      | `make models` | Saves machine learning models |
| `livetime`      | `make livetime` | Calculates and saves detector livetimes |
| `efficiencies`      | `make efficiencies` | Calculates and saves detection efficiencies |
| `plots`      | `make plots` | Makes all analysis plots |

*Note*: a `-s` option can be added to `make` to run commands in silent mode (i.e. doesn't print the commands as they are executed).

To reproduce this analysis run:

```bash
make processing-cluster
# Wait for cluster jobs to finish
make processing
make plots
```


## License

[MIT License](LICENSE)

Copyright (c) 2016-2018 James Bourbeau
