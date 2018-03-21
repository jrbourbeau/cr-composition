#!/bin/bash -e

ENV_PATH=${1:-"$PWD/.env"}
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Don't want PYTHONPATH environment variable set inside virtual environment
if [[ -z "${PYTHONPATH}" ]]; then
    echo ""
else
    echo -e "${ORANGE}Unsetting the PYTHONPATH environment variable."
    echo "PYTHONPATH is currently set to:"
    echo "      ${PYTHONPATH}"
    echo -e "${NC}"
    unset PYTHONPATH
fi

echo -e "${CYAN}Creating virtual environment at location $ENV_PATH...${NC}"
virtualenv --python=/cvmfs/icecube.opensciencegrid.org/py2-v3/RHEL_6_x86_64/bin/python --prompt="(cr-composition) " $ENV_PATH
source $ENV_PATH/bin/activate
echo ""

echo -e "${CYAN}pip version:${NC}"
pip --version
echo ""

echo -e "${CYAN}Installing Python dependencies...${NC}"
pip install svn+http://code.icecube.wisc.edu/svn/sandbox/james.bourbeau/PyUnfold
pip install -e .
echo ""

echo -e "${GREEN}Successfully installed all needed Python packages. To activate the environment use the command:"
echo "      source $ENV_PATH/bin/activate"
echo "Activating Python virtual environment at $ENV_PATH"
echo -e "${NC}"
source $ENV_PATH/bin/activate


echo -e "${CYAN}Getting ROOT v5.34.36 source code..."
echo -e "${NC}"
# Taken from https://root.cern.ch/get-root-sources
wget https://root.cern.ch/download/root_v5.34.36.source.tar.gz
tar -zxf root_v5.34.36.source.tar.gz
rm root_v5.34.36.source.tar.gz

echo -e "${CYAN}Building ROOT from source..."
echo -e "${NC}"
# Build ROOT from source
mkdir root_build
cd root_build
cmake -Dasimage=OFF ../root
cmake --build . -- -j10
source bin/thisroot.sh
cd ../
