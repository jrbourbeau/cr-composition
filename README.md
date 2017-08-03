# Cosmic-ray composition analysis

This repository is used to investigate the cosmic-ray composition spectrum using data collected by the IceCube South Pole Neutrino Observatory.

### Repository layout

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

While not generated with, this layout was inspired by the [Cookiecutter Data Science project](https://github.com/drivendata/cookiecutter-data-science).


### Installation

The installation steps for this project are designed to be relatively hassle-free. If possible, install this project in its own virtual Python environment.

1. Clone this repository to your local machine with
```bash
git clone https://github.com/jrbourbeau/cr-composition.git
```
This command will create a local copy of `cr-composition`.

2. Install the `comptools` Python package via `pip` with the following command:
```bash
pip install -e /path/to/cr-composition
```
This will install `comptools` along with all of the required dependencies (`pandas`, `numpy`, `scikit-learn`, etc.).
