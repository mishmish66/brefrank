# Use the base image with CUDA and CUDNN
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install some basic utilities
RUN apt update --fix-missing && \
    apt install -y wget git && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install Miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -a && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Update Conda
RUN conda update conda

# Copy environment.yml file to Docker
COPY environment.yml /tmp/environment.yml

# Create a Python 3.11 environment using environment.yml
RUN conda env update -n base -f /tmp/environment.yml

# Clean up
RUN conda clean --all --yes

# Set the default command to run on boot
CMD [ "/bin/bash" ]
