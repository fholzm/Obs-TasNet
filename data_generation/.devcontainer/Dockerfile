FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

RUN apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends --force-yes \
    build-essential \
    bzip2 \
    coreutils \
    curl \
    ca-certificates \
    git \
    jackd \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    htop \
    mercurial \
    openssh-client \
    procps \
    python3 \
    python3-venv \
    python3-pip \
    python-is-python3 \
    software-properties-common \
    subversion \
    wget \
    zsh \
    && apt-get clean

RUN wget -qO- http://apt.hoertech.de/openmha-packaging.pub | tee /etc/apt/trusted.gpg.d/openmha-packaging.asc
RUN apt-add-repository 'deb [arch=amd64] http://apt.hoertech.de noble universe'

RUN apt-get update -q
RUN apt-get install -y -q tascarpro


# Create and activate virtual environment
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH

# Upgrade pip inside the venv and install packages
RUN $VENV_PATH/bin/pip install --upgrade pip \
    && $VENV_PATH/bin/pip install lxml numpy pycodestyle scipy soundfile joblib tqdm colorednoise toml

# Ensure commands use the virtual environment by default
ENV PATH="$VENV_PATH/bin:$PATH"