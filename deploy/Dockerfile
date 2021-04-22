#/************************************************************************************#***
#***	Copyright Dell 2021, All Rights Reserved.
#***
#***	File Author: Dell, 2021年 04月 21日 星期三 13:36:22 CST
#***
#************************************************************************************/
#

# Runner docker is for deployment ...

# Dependencies: onnxruntime-gpu 1.6, cuda-10.2 (Driver version 440.95.01), cudnn8 and basic system files ...
# All these denpended files come from makefile pickers， self-sufficient !!!

FROM ubuntu:18.04
MAINTAINER Dell Du <18588220928@163.com>

# ENV DEBIAN_FRONTEND=noninteractive
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn

ENV CUDA_VERSION=10.2.89
ENV CUDA_PKG_VERSION=10-2=10.2.89-1 
ENV CUDNN_VERSION=8.0.5.39

WORKDIR /home

RUN apt-get update && \
	apt-get install -y --no-install-recommends gnupg2 curl wget ca-certificates openssh-client && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
	ssh -f -N -D 127.0.0.1:1080 ubuntu@23.248.162.51 && \
	apt-get -o Acquire::http::proxy="socks5h://127.0.0.1:1080/" update && \
    rm -rf /var/lib/apt/lists/*

# apt-get update
# apt-get install -y wget openssh-client net-tools gnupg2
# ssh -f -N -D 127.0.0.1:1080 ubuntu@23.248.162.51
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" >> /etc/apt/sources.list
# apt-get -o Acquire::http::proxy="socks5h://127.0.0.1:1080/" update

# Install CUDA 10.2
RUN apt-get update && apt-get install -y --no-install-recommends cuda-cudart-$CUDA_PKG_VERSION cuda-compat-10-2 && \
	ln -s cuda-10.2 /usr/local/cuda && \
	rm -rf /var/lib/apt/lists/*

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
	echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Install Cublas 10
RUN apt-get update && apt-get install -y --no-install-recommends cuda-libraries-$CUDA_PKG_VERSION libcublas10=10.2.2.89-1 && \
	rm -rf /var/lib/apt/lists/*

# Install cudnn8
RUN apt-get update && \
	apt-get install -y --no-install-recommends libcudnn8=$CUDNN_VERSION-1+cuda10.2 && \
	apt-mark hold libcudnn8 && \
	rm -rf /var/lib/apt/lists/*

# 1. Setup building tools ...
# Command "apt-get" clean need apt-utils package
# Delete nvidia update for security !!!
# RUN rm -rf /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/cuda.list

# RUN apt-get -y update && \
# 	apt-get install -y \
# 	apt-utils \
# 	gcc g++ \
# 	make \
# 	git && \
# 	apt-get clean

# RUN apt-get install -y --no-install-recommends cuda-10-2 cuda-cudart-10-2 libcudnn8 libcublas10 libcublas10 && \
# 	apt-get clean

# 2. Add onnxruntime libs ...
ADD temp/ /


# EXPOSE 9101 9102 9103 9104 9105 9106 9107 9108 9109 9201 9202 9203
