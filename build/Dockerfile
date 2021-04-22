#/************************************************************************************#***
#***	Copyright Dell 2021, All Rights Reserved.
#***
#***	File Author: Dell, 2021年 04月 21日 星期三 13:36:22 CST
#***
#************************************************************************************/
#

# Building docker is for developers ...

# Dependencies: NImage, onnxruntime-gpu 1.6, cuda-10.2 (Driver version 440.95.01), cudnn8

FROM nvidia/cuda:10.2-cudnn8-runtime
MAINTAINER Dell Du <18588220928@163.com>

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

WORKDIR /build_workspace

# 1. Setup building tools ...
# Command "apt-get" clean need apt-utils package
# Delete nvidia update for security !!!
RUN rm -rf /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/cuda.list

RUN apt-get -y update && \
	apt-get install -y \
	apt-utils \
	gcc g++ \
	make \
	git && \
	apt-get clean

# 2. Add onnxruntime libs ...
ADD temp/onnxruntime/ /opt/onnxruntime/

# 3. Building NImage
# NImage need packages ...
RUN apt-get install -y libpng-dev libjpeg-dev libnanomsg-dev zlib1g-dev && \
	apt-get clean

RUN git clone https://github.com/delldu/NImage.git
RUN cd NImage && \
	make -C msgpackc && make -C msgpackc install && make -C msgpackc clean && \
	make && make install && make clean

# 4. Clean installed packages ...
RUN apt-get autoremove

EXPOSE 9101 9102 9103 9104 9105 9106 9107 9108 9109 9201 9202 9203
