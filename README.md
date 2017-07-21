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



### Installation

Clone this repository
```bash
git clone https://github.com/jrbourbeau/cr-composition.git
```

The `comptools` Python package, can be installed via `pip` with the following command:
```bash
pip install -e /path/to/cr-composition
```
