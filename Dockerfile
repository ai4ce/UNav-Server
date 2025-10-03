FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# 1. Install basic Linux utilities and required C++ libraries (Eigen3, Ceres, CMake)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        git \
        vim \
        cmake \
        libeigen3-dev \
        libceres-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Miniconda to /opt/conda and update PATH
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# 3. Copy Conda environment file
COPY environment.yml /tmp/environment.yml

# 4. Create Conda environment 'unav' and clean up
# 先接受 Anaconda TOS（defaults/r 渠道）
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda env create -f /tmp/environment.yml && conda clean -afy


# Set default shell to use 'conda run' for subsequent RUN instructions
SHELL ["conda", "run", "-n", "unav", "/bin/bash", "-c"]

# 5. Set working directory and copy all project files
WORKDIR /workspace
COPY . /workspace
RUN rm -f /workspace/config.py

# 6. Install external/private Python packages (no dependency resolution)
RUN pip install --no-deps git+https://github.com/cvg/implicit_dist.git
RUN pip install --no-deps --upgrade git+https://github.com/endeleze/UNav.git

# 7. Expose the API port
EXPOSE 5001

# 8. Set entrypoint: activate environment and launch FastAPI app via Uvicorn
CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate unav && uvicorn main:app --host 0.0.0.0 --port 5001 --log-level info"]