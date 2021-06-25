FROM nvidia/cuda:11.2.1-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    vim
WORKDIR /opt

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh

ENV PATH /opt/miniconda3/bin:$PATH

COPY jukebox.yml .

RUN pip install --upgrade pip && \
    conda update -n base -c defaults conda && \
    conda env create -n jukebox -f jukebox.yml && \
    conda init && \
    echo "conda activate jukebox" >> ~/.bashrc
    
ENV CONDA_DEFAULT_ENV jukebox && \
    PATH /opt/conda/envs/jukebox/bin:$PATH

WORKDIR /workspace

CMD ["/bin/bash"]


