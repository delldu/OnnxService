#/************************************************************************************
#***
#***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2021-02-07 17:38:31
#***
#************************************************************************************/
#
INSTALL_DIR=/usr/local
ONNXMODEL_INSTALL_DIR=/opt/onnxservice
ONNXRUNTIME_INSTALL_DIR=/opt/onnxruntime

XSUBDIRS :=  \
	lib \
	image_nima \
	image_zoom \
	image_patch \
	image_color \
	image_clean \
	image_light \
	image_matting \
	video_clean \
	video_color \
	video_slow

all:
	@for d in $(XSUBDIRS)  ; do \
		if [ -d $$d ] ; then \
			$(MAKE) -C $$d || exit 1; \
		fi \
	done	

install:
	@for d in $(XSUBDIRS)  ; do \
		if [ -d $$d ] ; then \
			$(MAKE) -C $$d install || exit 1; \
		fi \
	done	

clean:
	@for d in $(XSUBDIRS) ; do \
		if [ -d $$d ] ; then \
			$(MAKE) -C $$d clean || exit 1; \
		fi \
	done	
