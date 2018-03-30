.. cr-composition documentation master file, created by
   sphinx-quickstart on Mon Feb 12 18:24:22 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/jrbourbeau/cr-composition

.. toctree::
   :maxdepth: 3
   :caption: User Guide:
   :hidden:

   Overview <self>
   installation
   comptools

IceCube cosmic-ray composition analysis
=======================================


.. image:: https://travis-ci.org/jrbourbeau/cr-composition.svg?branch=master
    :target: https://travis-ci.org/jrbourbeau/cr-composition

.. image:: https://img.shields.io/badge/python-2.7-blue.svg
    :target: https://github.com/jrbourbeau/cr-composition


This repository has the analysis code used to investigate the cosmic-ray composition spectrum using data collected by the IceCube South Pole Neutrino Observatory.


Repository layout
-----------------

The directory structure of this analysis is given by:

.. code-block:: bash

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


While not generated with, this layout for this project was inspired by the
`Cookiecutter Data Science project <https://github.com/drivendata/cookiecutter-data-science>`_.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
