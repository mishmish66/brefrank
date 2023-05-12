#!/bin/bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -u -p $HOME/miniconda

echo "Conda installed. Setting up bash"
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
echo "Initializing Bash"
~/miniconda/bin/conda init bash

# Source the changes made by `conda init bash`
source ~/.bashrc

apt install libglfw3-dev -y

echo "Activating Base"
conda activate base

conda install python=3.11 -y

# Install Jupyter Notebook using pip
pip install notebook

jupyter notebook --no-browser --NotebookApp.allow_origin='*' --port 8081 --allow-root --ip='*'