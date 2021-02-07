#/************************************************************************************
#***
#***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2021-02-07 16:51:15
#***
#************************************************************************************/
#
ONNXRUNTIME_INSTALL_DIR := /opt/onnxruntime

LIB_NAME := libonnxeng

INCS	:= -I$(ONNXRUNTIME_INSTALL_DIR)/include $(shell nimagetool --cflags)

SOURCE :=  \
	engine.c

DEFINES := -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++14 -std=gnu++14 

CXXFLAGS := -fPIC -Wall -Wextra
LDFLAGS := -fPIC -O3


#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
CFLAGS   := ${CFLAGS} ${DEFINES}
CXXFLAGS := ${CXXFLAGS} ${DEFINES}
OBJECTS := $(addsuffix .o,$(basename ${SOURCE}))

#****************************************************************************
# Compile block
#****************************************************************************
all: staticlib

staticlib:$(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB_NAME).a $(OBJECTS)


#****************************************************************************
# Depend block
#****************************************************************************
depend:

#****************************************************************************
# common rules
#****************************************************************************
%.o : %.c
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@


clean:
	rm -rf *.a *.so *.o $(OBJECTS)

