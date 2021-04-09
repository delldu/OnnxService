/************************************************************************************
***
***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2021-01-13 21:19:01
***
************************************************************************************/

#ifndef _ENGINE_H
#define _ENGINE_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <onnxruntime_c_api.h>
#include <nimage/image.h>
#include <nimage/nnmsg.h>

// ONNX Runtime Engine
typedef struct {
	DWORD magic;
	const char *model_path;
	int use_gpu;
	OrtEnv *env;
	OrtSession *session;
	OrtSessionOptions *session_options;

	std::vector < const char *>input_node_names;
	std::vector < const char *>output_node_names;
} OrtEngine;

#define CheckEngine(e) \
    do { \
            if (! ValidEngine(e)) { \
				fprintf(stderr, "Bad OrtEngine.\n"); \
				exit(1); \
            } \
    } while(0)

OrtEngine *CreateEngine(const char *model_path, int use_gpu);
int ValidEngine(OrtEngine * engine);
TENSOR *TensorForward(OrtEngine * engine, TENSOR * input);
void DestroyEngine(OrtEngine * engine);

int OnnxService(char *endpoint, char *onnx_file, int use_gpu);

TENSOR *OnnxRPC(int socket, TENSOR * input, int reqcode, int *rescode);
TENSOR *ResizeOnnxRPC(int socket, TENSOR *send_tensor, int reqcode, int *rescode, int multiples);
TENSOR *ZeropadOnnxRPC(int socket, TENSOR *send_tensor, int reqcode, int *rescode, int multiples);

void SaveOutputImage(IMAGE *image, char *filename);
void SaveTensorAsImage(TENSOR *tensor, char *filename);


#endif // _ENGINE_H
