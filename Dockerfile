FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# 1. Install basic Linux utilities and required C++ libraries (Eigen3, Ceres, CMake)
# Also install python3-pip for system Python (required by Modal runtime for protobuf)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        git \
        vim \
        cmake \
        libeigen3-dev \
        libceres-dev \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Modal runtime deps for system Python.
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED && pip3 install protobuf grpclib

# 2. Install Miniconda to /opt/conda and update PATH
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Modal runtime resolves "python" from PATH; with conda first in PATH,
# core Modal deps must exist in conda base.
RUN /opt/conda/bin/pip install --no-cache-dir protobuf grpclib

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

# Install protobuf for Modal SDK runtime (Modal injects its own Python entrypoint)
RUN pip install protobuf

# Fail image build early if Modal bootstrap deps are visible.
RUN /opt/conda/bin/python -c "from google.protobuf.empty_pb2 import Empty; from grpclib import Status; print('modal deps import OK')"

# 7. Expose the API port
EXPOSE 5001

# 8. Set entrypoint: activate environment and launch FastAPI app via Uvicorn
CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate unav && uvicorn main:app --host 0.0.0.0 --port 5001 --log-level info"]
