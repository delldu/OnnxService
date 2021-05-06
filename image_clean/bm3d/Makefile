#/************************************************************************************
#***
#***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2020-11-20 16:48:15
#***
#************************************************************************************/
#
INSTALL_DIR := /usr/local

CUDA_INSTALL_DIR := /usr/local/cuda-10.2
CUDA_INCLUDE := ${CUDA_INSTALL_DIR}/include
NVCC := ${CUDA_INSTALL_DIR}/bin/nvcc

LIB_NAME := libbm3d

CUDA_FLAGS= -arch=sm_61

INCS := -I${CUDA_INCLUDE}
DEFINES := 
CXXFLAGS := -O2 -fPIC -Wall -Wextra --std=c++11

#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
CXXFLAGS := ${CXXFLAGS} ${DEFINES}
OBJECTS := bm3d.o filtering.o blockmatching.o dct8x8.o

#****************************************************************************
# Compile block
#****************************************************************************
all: staticlib

staticlib:$(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB_NAME).a $(OBJECTS)

#****************************************************************************
# Depend block
#****************************************************************************
bm3d.o: bm3d.cpp bm3d_implement.cpp
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

filtering.o: filtering.cu params.hpp
	@${NVCC} $(addprefix -I,$(CUDA_INCLUDE)) -m64 -c $(CUDA_FLAGS) $< -o $@

blockmatching.o: blockmatching.cu params.hpp
	@${NVCC} $(addprefix -I,$(CUDA_INCLUDE)) -m64 -c --std=c++11 $(CUDA_FLAGS) $< -o $@

dct8x8.o: dct8x8.cu
	@${NVCC} $(addprefix -I,$(CUDA_INCLUDE)) -m64 -c $(CUDA_FLAGS) $< -o $@

install:
	cp $(LIB_NAME).a ${INSTALL_DIR}/lib
	cp bm3d.h ${INSTALL_DIR}/include

clean:
	rm -rf *.a *.so *.o $(OBJECTS)

