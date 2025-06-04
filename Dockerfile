FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# 1. Install basic tools and C++ libraries (Eigen3, Ceres, CMake)
RUN apt-get update && \
    apt-get install -y wget git vim cmake libeigen3-dev libceres-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# 3. Copy Conda environment specification
COPY environment.yml /tmp/environment.yml

# 4. Create and activate the 'unav' conda environment
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# NOTE: The following SHELL affects only RUN, not CMD/ENTRYPOINT
# It's recommended to use `conda run ...` explicitly in the final command.
SHELL ["conda", "run", "-n", "unav", "/bin/bash", "-c"]

# 5. Set working directory and copy project code
WORKDIR /workspace
COPY . /workspace

# 6. Install private local package (implicit_dist) and GitHub package (ai4ce/unav) using a build argument for GitHub token
# The GITHUB_TOKEN argument should be passed at build time for security.
RUN pip install --no-deps ./third_party/unav \
 && pip install ./third_party/implicit_dist

# 7. Expose port
EXPOSE 5001

# 8. Entry point
# Use exec form to avoid zombie processes
CMD ["conda", "run", "-n", "unav", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
