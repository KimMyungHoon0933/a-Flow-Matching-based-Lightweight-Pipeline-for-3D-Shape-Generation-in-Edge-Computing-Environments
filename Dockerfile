FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt install libopencv-dev software-properties-common -y
RUN apt install python3-pip python3.10-dev -y
RUN pip3 install --upgrade pip && pip3 install setuptools==68.0.0 && pip3 install torch==2.1.0 ninja

WORKDIR /root
ADD requirements.txt .
RUN pip3 install --ignore-installed blinker && pip3 install -r requirements.txt

## install external packages
ENV TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ADD external/pointops .
ADD external/pointops /root/external/pointops
RUN pip3 install /root/external/pointops

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
WORKDIR /workspace