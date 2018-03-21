FROM ubuntu:14.04

WORKDIR /root/jbourbeau
ENV HOME=/root/jbourbeau

# Install dependencies needed to build icerec
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libbz2-dev \
    libgl1-mesa-dev \
    freeglut3-dev \
    libxml2-dev \
    subversion \
    libboost-python-dev \
    libboost-system-dev \
    libboost-signals-dev \
    libboost-thread-dev \
    libboost-date-time-dev \
    libboost-serialization-dev \
    libboost-filesystem-dev \
    libboost-program-options-dev \
    libboost-regex-dev \
    libboost-iostreams-dev \
    libgsl0-dev \
    libcdk5-dev \
    libarchive-dev \
    python-scipy \
    ipython-qtconsole \
    libqt4-dev \
    python-urwid \
    # Extra stuff for newer Ubuntu versions (>=14.04)
    libz-dev \
    libqt5opengl5-dev \
    libstarlink-pal-dev \
    python-sphinx \
    libopenblas-dev \
    # Simulation packages
    libcfitsio3-dev \
    libsprng2-dev \
    libmysqlclient-dev \
    libsuitesparse-dev \
    # IceRec Packages
    libcfitsio3-dev \
    libmysqlclient-dev \
    libhdf5-serial-dev \
    # ROOT
    root-system

# Install other stuff
RUN apt-get install -y \
    zsh \
    git-core \
    python-pip

# Install oh-my-zsh
RUN git clone git://github.com/robbyrussell/oh-my-zsh.git $HOME/.oh-my-zsh \
     && cp $HOME/.oh-my-zsh/templates/zshrc.zsh-template $HOME/.zshrc

# Upgrade pip
RUN pip install --upgrade pip \
    && pip --version \
    && pip list

# Install comptools
COPY . $HOME/cr-composition
RUN pip install -e $HOME/cr-composition

# # Check out icerec from SVN
# RUN mkdir $HOME/icerec \
#    && mkdir $HOME/icerec/build \
#    && svn co http://code.icecube.wisc.edu/svn/meta-projects/icerec/releases/V05-01-00 \
#           $HOME/icerec/src --username=icecube --password=skua --no-auth-cache

# Build icerec
# WORKDIR $HOME/icerec/build
# RUN cmake $HOME/icerec/src \
#    && make

# Provide the entry point to run commands
#ENTRYPOINT ["/bin/bash", "$HOME/icerec/build/env-shell.sh", "exec"]
CMD ["/usr/bin/zsh"]
