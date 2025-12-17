FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Basic utilities (as root)
RUN apt-get update \
    && apt-get install -y sudo wget rsync rclone git tmux build-essential cmake dssp zip nano unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

## Install Miniconda (as root)
ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update \
    && apt-get install -y wget git \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O miniconda.sh -q && \
    mkdir -p /opt && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Accept Anaconda ToS for required channels (non-interactive builds)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

## Create conda env
ARG ENV_NAME="tissuevit"

# Copy env and requirements
COPY environment_verbose.yaml .
COPY requirements.txt .
COPY tissuevit.requirements.txt .

# Create the environment (as root), install requirements and install packages that need special flags (those packages can't be installed directly via requirements.txt)
RUN conda env create --yes -f environment_verbose.yaml -n ${ENV_NAME} && \
    conda run -n ${ENV_NAME} pip install -r requirements.txt && \
    conda run -n ${ENV_NAME} pip install -r tissuevit.requirements.txt && \
    conda run -n ${ENV_NAME} pip install "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "torchaudio==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124 && \
    conda run -n ${ENV_NAME} pip install "xformers==0.0.29.post3" --index-url https://download.pytorch.org/whl/cu124 && \
    conda run -n ${ENV_NAME} pip install --no-cache-dir "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" && \
    conda run -n ${ENV_NAME} pip install "torch-scatter==2.1.2+pt26cu124" -f https://data.pyg.org/whl/torch-2.6.0+cu124.html && \
    conda clean -afy && \
    chmod 777 -R /opt/conda

# Ensure conda is available in non-interactive shells and auto-activate env
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate ${ENV_NAME}" >> /root/.bashrc && \
    mkdir -p /etc/skel && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/skel/.bashrc && \
    echo "conda activate ${ENV_NAME}" >> /etc/skel/.bashrc

# Install libvips (as root)
RUN apt-get update \
    && apt-get install libvips-dev --no-install-recommends -y

# Use bash with conda support for subsequent RUNs
SHELL ["/bin/bash", "-ic"]

# LDAP / user settings
ARG LDAP_USERNAME="gotti"
ARG LDAP_UID=226403
ARG LDAP_GROUPNAME="rcp-runai-course-cs-433-group08"
ARG LDAP_GID=86764

# Password is "12345"
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID} && \
    useradd -m -p '$6$v45mq2mleqYge2AX$2CT7LFbRPup9aOrMLPC4Kmwb8dbXVogmYBPCMY1QnW8cmQDvsV7EatU9Gw6N5M4eFEmqabPG0cHNClW3KDFtL.' -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME} && \
    usermod -aG sudo ${LDAP_USERNAME} && \
    echo "${LDAP_USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${LDAP_USERNAME}

# Switch to the local user
USER ${LDAP_USERNAME}

# Workdir and PATH
ARG WORKDIR="/home/${LDAP_USERNAME}"
WORKDIR ${WORKDIR}

RUN mkdir -p ${WORKDIR}/bin
ENV PATH="${WORKDIR}/bin:/opt/conda/envs/${ENV_NAME}/bin:/opt/conda/bin:${PATH}"
