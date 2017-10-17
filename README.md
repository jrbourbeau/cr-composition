# Cosmic-ray composition analysis

[![Build Status](https://travis-ci.org/jrbourbeau/cr-composition.svg?branch=master)](https://travis-ci.org/jrbourbeau/cr-composition)

This repository is used to investigate the cosmic-ray composition spectrum
using data collected by the IceCube South Pole Neutrino Observatory.


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
Python environment.

1. Clone this repository to your local machine with
   ```bash
   git clone https://github.com/jrbourbeau/cr-composition.git
   ```
   This command will create a local copy of `cr-composition`.

2. Install the `comptools` Python package via `pip` with the following command:
   ```bash
   pip install -e /path/to/cr-composition
   ```
   This will install `comptools` along with all of the required dependencies
   (`pandas`, `numpy`, `scikit-learn`, etc.).

3. Install version 05-01-00 of the icerec IceCube metaproject.


## Quickstart

The steps needed to run this analysis are listed below. The details of these
commands are stored in `Makefile`. (*Note: the following commands should be
run from the main `cr-composition` directory*)

1. Ensure that you have the appropriate computing environment (i.e. you've
   run the `env-shell.sh` script for your icerec metaproject and activated
   the virtual environment in which the `comptools` package has been installed).

2. Process simulation and data by running `make simulation` and `make data`.
   These commands will try to submit jobs to HTCondor, so make sure to run
   them on a condor submitting machine. In order to obey the best practices for
   submitting condor jobs, you should run `make data` only after the
   simulation jobs have completed (this avoids having too many DAGs running at
   the same time).

3. Calculate and save detector livetimes by running `make save-livetimes`.

4. (*in progress*) Calculate and save detector effective areas by running
   `make save-effective-areas`.

5. (*in progress*) All analysis plots can now be made with `make plots`.
