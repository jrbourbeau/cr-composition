#!/bin/bash -e

HERE="$PWD"

VENV_PATH="$PWD/venv"
METAPROJECT_PATH="$PWD/metaproject"

function usage()
{
    echo "if this was a real script you would see something useful here"
    echo ""
    echo "./simple_args_parsing.sh"
    echo "\t-h --help"
    echo "\t--venv-path=$VENV_PATH"
    echo "\t--metaproject-path=$METAPROJECT_PATH"
    echo ""
}

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        --venv-path)
            VENV_PATH=$VALUE
            ;;
        --metaproject-path)
            METAPROJECT_PATH=$VALUE
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

# Setup color printing!
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}Creating virtual Python environment in $VENV_PATH \n${NC}"
echo -e "${CYAN}Creating IceCube metaproject in $METAPROJECT_PATH \n${NC}"


#######################
# Set up py2-v3 toolset
#######################

echo -e "${CYAN}Setting up py2-v3 toolset...\n${NC}"
TOOLSET=py2-v3
CVMFS_ROOT=/cvmfs/icecube.opensciencegrid.org
eval `$CVMFS_ROOT/$TOOLSET/setup.sh`
# ROOT 6 issue
export PYTHONPATH="/cvmfs/icecube.opensciencegrid.org/py2-v3/RHEL_6_x86_64/lib:$PYTHONPATH"
# Add gcc version 4.8 to path
source /opt/rh/devtoolset-2/enable


###################################
# Create Python virtual environment
###################################

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

echo -e "${CYAN}Creating virtual environment at location $VENV_PATH...${NC}"
virtualenv --prompt="(cr-composition) " "$VENV_PATH"
source "$VENV_PATH"/bin/activate
echo ""

echo -e "${CYAN}pip version:${NC}"
pip --version
echo ""

###########################
# Install comptools package
###########################
echo -e "${CYAN}Installing Python dependencies...${NC}"
# pip install -r requirements.txt
# pip install svn+http://code.icecube.wisc.edu/svn/sandbox/james.bourbeau/PyUnfold
pip install --process-dependency-links -e .
echo ""

echo -e "${GREEN}Successfully installed all needed Python packages. To activate the environment use the command:"
echo "      source $VENV_PATH/bin/activate"
echo "Activating Python virtual environment at $VENV_PATH"
echo -e "${NC}"


###########################
# Create icerec metaproject
###########################
echo -e "${CYAN}Setting up icerec metaproject...${NC}"
mkdir "${METAPROJECT_PATH}"
echo -e "${CYAN}Checking out icerec V05-01-05 from the IceCube SVN into ${METAPROJECT_PATH}/src ${NC}"
svn co --username=icecube --password=skua --no-auth-cache http://code.icecube.wisc.edu/svn/meta-projects/icerec/releases/V05-01-05 "${METAPROJECT_PATH}/src"
cd "$METAPROJECT_PATH"
# cp -r /data/user/jbourbeau/metaprojects/icerec/V05-01-00/src .
mkdir -p "$METAPROJECT_PATH/build"
cd "$METAPROJECT_PATH/build"
echo -e "${CYAN}Using cmake to create Makefiles${NC}"
cmake -DCMAKE_CXX_STANDARD=11 ../src
echo -e "${CYAN}Building icerec metaproject${NC}"
make -j5
echo -e "${GREEN}Icerec metaproject successfully built!!!"
echo -e "${GREEN}To activate the metaproject run:"
echo "      $METAPROJECT_PATH/build/env-shell.sh"
echo -e "${NC}"

cd "${HERE}"
