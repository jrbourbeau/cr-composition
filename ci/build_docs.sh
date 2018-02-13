#!/usr/bin/env bash

set -e

echo "Building documentation..."
pip install sphinx numpydoc sphinx_rtd_theme
cd docs
make clean
make html
cd ../
echo "Successfully built the documentation!"
