# Prebuilt toolchain for Linux CI jobs.
#
# The image is published by .github/workflows/build-ci-image.yml and is
# intended to replace the per-job apt-get installs in the workflows.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends \
        -o Acquire::Retries=5 \
        build-essential \
        ca-certificates \
        cargo \
        curl \
        cython3 \
        debhelper \
        devscripts \
        dh-python \
        dput \
        equivs \
        file \
        gettext \
        git \
        gnupg2 \
        jq \
        libfuse2t64 \
        libgl1 \
        libglib2.0-0 \
        patchelf \
        pkg-config \
        pybuild-plugin-pyproject \
        python3 \
        python3-all \
        python3-dev \
        python3-maturin \
        python3-numpy \
        python3-pip \
        python3-poetry-core \
        python3-pyclipper \
        python3-setuptools \
        python3-venv \
        python3-wheel \
        rsync \
        rustc \
        sudo \
        unzip \
        wget \
        xvfb \
        xz-utils \
        zip \
    && rm -rf /var/lib/apt/lists/*
