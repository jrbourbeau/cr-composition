# Cosmic-ray mass composition analysis

[![Build Status](https://travis-ci.org/jrbourbeau/cr-composition.svg?branch=master)](https://travis-ci.org/jrbourbeau/cr-composition)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)


This repository has the analysis code used to investigate the cosmic-ray composition spectrum using data collected by the IceCube South Pole Neutrino Observatory.


## Repository layout

The directory structure of this analysis is given by:

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make plots`
├── README.md          <- The top-level README for developers using this project.
├── data               <- Directory containing processed data/simulation
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks used for exploratory analysis
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── comptools          <- Python package for code used by multiple scripts.
│
├── processing         <- Scripts related to processing and saving simulation and data
│
└── plotting           <- Scripts related to making plots for analysis.
```

While not generated with, this layout for this project was inspired by the
[Cookiecutter Data Science project](https://github.com/drivendata/cookiecutter-data-science).


## Installation

The installation steps for this project are designed to be relatively
hassle-free. It's recommended to install this project in its own virtual
Python environment (e.g. using `virtualenv`).

1. Clone this repository to your local machine with
   ```bash
   git clone https://github.com/jrbourbeau/cr-composition.git
   ```
   This command will create a local copy of `cr-composition`.

2. Install the `comptools` Python package with the following command:
   ```bash
   pip install -e cr-composition
   ```
   This will install `comptools` along with all of the required dependencies
   (`pandas`, `numpy`, `scikit-learn`, etc.).

3. Install version `V05-01-05` of the icerec IceCube metaproject.


## Quickstart

The steps needed to run this analysis are listed below. (*Note*: the following commands should be
run from inside the `cr-composition` directory)

1. Ensure that you have the appropriate computing environment (i.e. you've run the `env-shell.sh` script for your icerec metaproject and activated the virtual environment in which the `comptools` package has been installed).

2. Process simulation and data by running `make processing` (for a list of available commands see below). This will submit jobs to [HTCondor](https://research.cs.wisc.edu/htcondor/), so make sure to run them on a machine that has `condor_submit_dag`.

3. (*in progress*) Run `make analysis` to run the full analysis chain.


## Analysis steps

The workflow for this analysis is managed using GNU `make`. Various commands are stored in `Makefile` and can be executed using the syntax `make -s <command>`. The available commands in `Makefile` are shown below

| Name        | Command           | Description  |
|:-------------:|:-------------:| :-----|
| `sim`      | `make -s sim` | Processes and saves simulation data |
| `data`      | `make -s data` | Processes and saves data |
| `processing`      | `make -s processing` | Processes and saves both simulation and data |
| `livetimes`      | `make -s livetimes` | Calculates and saves detector livetimes |
| `efficiencies`      | `make -s efficiencies` | Calculates and saves detection efficiencies |
| `energy-reco`      | `make -s energy-reco` | Saves energy reconstruction models |
| `plots`      | `make -s plots` | Makes all analysis plots |
| `analysis`      | `make -s analysis` | Runs all commands to reproduce full analysis |

*Note*: the (optional) `-s` option runs `make` in silent mode (i.e. doesn't print the commands as they are executed).
