#!/usr/bin/env bash

ENV_PATH=${1:-"$PWD/.env"}

echo "Creating virtual environment at location $ENV_PATH..."
virtualenv --python=/cvmfs/icecube.opensciencegrid.org/py2-v3/RHEL_6_x86_64/bin/python --prompt="(cr-composition) " $ENV_PATH
source $ENV_PATH/bin/activate
echo ""

echo "pip version:"
pip --version
echo ""

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo ""

echo "Successfully created the virtual environment for this analysis!"
echo ""
echo "To activate the environment use the command:"
echo "      source $ENV_PATH/bin/activate"
echo ""
