# Unfolding using PyUnfold

- `run_unfolding.py` &mdash; Python script to perform an iterative Bayesian unfolding using PyUnfold. Will save unfolded distributions (and errors) to `pyunfold_output.hdf` in the current directory

- `config.cfg` &mdash; Input config file for `run_unfolding.py` (I'd like to get rid of the need for this soon)

- `make_pyunfold_input_file.py` &mdash; Script to create input .root file for `run_unfolding.py`. Will generate a file `pyunfold_input.root` in the currrent directory (would, ideally, like to get rid of the need for this as well)
