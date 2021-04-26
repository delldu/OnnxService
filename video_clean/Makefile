#/************************************************************************************
#***
#***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2020-11-22 02:26:07
#***
#************************************************************************************/
#
INSTALL_DIR := $(shell grep INSTALL_DIR ../Makefile | cut -d '=' -f 2)
ONNXRUNTIME_INSTALL_DIR := $(shell grep ONNXRUNTIME_INSTALL_DIR ../Makefile | cut -d '=' -f 2)

TARGET := video_clean

SOURCE :=  \
	main.c \

INCS	:= -I../lib -I$(ONNXRUNTIME_INSTALL_DIR)/include $(shell nimagetool --cflags)

LIBS	:= -L../lib -lonnxeng -L$(ONNXRUNTIME_INSTALL_DIR)/lib -lonnxruntime \
		$(shell nimagetool --libs) \
		-Wl,-rpath=$(ONNXRUNTIME_INSTALL_DIR)/lib

DEFINES := -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++14 -std=gnu++14

CXXFLAGS := -O2 -fPIC -Wall -Wextra
LDFLAGS := -fPIC

#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
CXXFLAGS := ${CXXFLAGS} ${DEFINES}

OBJECT := $(addsuffix .o,$(basename ${SOURCE}))

#****************************************************************************
# Compile block
#****************************************************************************
all: ${TARGET}

${TARGET}: ${OBJECT}
	${CXX} ${LDFLAGS} -o ${TARGET} ${OBJECT} ${LIBS}

install:
	install -m 755 ${TARGET} ${INSTALL_DIR}/bin
	cp video_clean.service /lib/systemd/system/
	systemctl daemon-reload


#****************************************************************************
# common rules
#****************************************************************************
%.o : %.c
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

#****************************************************************************
# Depend block
#****************************************************************************
depend:

clean:
	rm -f core ${OBJECT} ${TARGET}
