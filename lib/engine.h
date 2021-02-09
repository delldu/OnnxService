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

void CheckStatus(OrtStatus * status);

OrtValue *CreateOrtTensor(TENSOR * tensor, int gpu);
int ValidOrtTensor(OrtValue * tensor);
size_t OrtTensorDimensions(OrtValue * tensor, int64_t * dims);
float *OrtTensorValues(OrtValue * tensor);
void DestroyOrtTensor(OrtValue * tensor);

OrtEngine *CreateEngine(const char *model_path, int use_gpu);
int ValidEngine(OrtEngine * engine);
OrtValue *SimpleForward(OrtEngine * engine, OrtValue * input_tensor);
TENSOR *TensorForward(OrtEngine * engine, TENSOR * input);
void DestroyEngine(OrtEngine * engine);

int OnnxService(char *endpoint, char *onnx_file, int use_gpu);
TENSOR *OnnxRPC(int socket, TENSOR * input, int reqcode, float option, int *rescode);

#endif							// _ENGINE_H
