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

#define IMAGE_CLEAN_SERVICE 0x01010000
#define IMAGE_CLEAN_SERVICE_WITH_GUIDED_FILTER 0x01010001
#define IMAGE_CLEAN_SERVICE_WITH_BM3D 0x01010002

#define IMAGE_CLEAN_URL "tcp://127.0.0.1:9101"

#define IMAGE_COLOR_SERVICE 0x01020000
#define IMAGE_COLOR_URL "tcp://127.0.0.1:9102"

#define IMAGE_NIMA_SERVICE 0x01030000
#define IMAGE_NIMA_URL "tcp://127.0.0.1:9103"

#define IMAGE_PATCH_SERVICE 0x01040000
#define IMAGE_PATCH_URL "tcp://127.0.0.1:9104"

#define IMAGE_ZOOM_SERVICE 0x01050000
#define IMAGE_ZOOM_SERVICE_WITH_PAN 0x01050001
#define IMAGE_ZOOM_URL "tcp://127.0.0.1:9105"

#define IMAGE_LIGHT_REQCODE 0x01060000
#define IMAGE_LIGHT_URL "tcp://127.0.0.1:9106"

#define IMAGE_MATTING_SERVICE 0x01070000
#define IMAGE_MATTING_URL "tcp://127.0.0.1:9107"

#define IMAGE_FACEGAN_SERVICE 0x01080000
#define IMAGE_FACEGAN_URL "tcp://127.0.0.1:9108"

#define VIDEO_CLEAN_SERVICE 0x02010000
#define VIDEO_CLEAN_URL "tcp://127.0.0.1:9201"

#define VIDEO_COLOR_SERVICE 0x02020000
#define VIDEO_REFERENCE_SERVICE 0x02020001
#define VIDEO_COLOR_URL "tcp://127.0.0.1:9202"

#define VIDEO_SLOW_SERVICE 0x02030000
#define VIDEO_SLOW_URL "tcp://127.0.0.1:9203"

// ONNX Runtime Engine
typedef struct {
	DWORD magic;
	const char *model_path;
	int use_gpu;
	OrtEnv *env;
	OrtSession *session;
	OrtSessionOptions *session_options;

	// Last i/o node dims
	int64_t input_node_dims[4];
	int64_t output_node_dims[4];

	std::vector < const char *>input_node_names;
	std::vector < const char *>output_node_names;
} OrtEngine;

// typedef int (*CustomSevice)(int socket, int service_code, TENSOR *input_tensor);
typedef int (*CustomSevice)(int, int, TENSOR *);

#define CheckEngine(e) \
    do { \
            if (! ValidEngine(e)) { \
				fprintf(stderr, "Bad OrtEngine.\n"); \
				exit(1); \
            } \
    } while(0)

OrtEngine *CreateEngine(char *model_path, int use_gpu);
OrtEngine *CreateEngineFromArray(void* model_data, size_t model_data_length, int use_gpu);

int ValidEngine(OrtEngine * engine);
TENSOR *TensorForward(OrtEngine * engine, TENSOR * input);
void DumpEngine(OrtEngine * engine);
void DestroyEngine(OrtEngine * engine);

int OnnxService(char *endpoint, char *onnx_file, int service_code, int use_gpu, CustomSevice custom_service_function);
int OnnxServiceFromArray(char *endpoint, void* model_data, size_t model_data_length, int service_code, int use_gpu, CustomSevice custom_service_function);

TENSOR *OnnxRPC(int socket, TENSOR * input, int reqcode);
TENSOR *ResizeOnnxRPC(int socket, TENSOR *send_tensor, int reqcode, int multiples);
TENSOR *ZeropadOnnxRPC(int socket, TENSOR *send_tensor, int reqcode, int multiples);

void SaveOutputImage(IMAGE *image, char *filename);
void SaveTensorAsImage(TENSOR *tensor, char *filename);
int CudaAvailable();

#define ENGINE_IDLE_TIME (120*1000)	// 120 seconds
static TIME engine_last_running_time = 0;

#define StartEngine(engine, onnx_file, use_gpu) \
do { \
	if (engine == NULL) \
		engine = CreateEngine(onnx_file, use_gpu); \
	CheckEngine(engine); \
	engine_last_running_time = time_now(); \
} while(0)

#define StartEngineFromArray(engine, model_data, model_data_length, use_gpu) \
do { \
	if (engine == NULL) { \
		engine = CreateEngineFromArray(model_data, model_data_length, use_gpu); \
	} \
	CheckEngine(engine); \
	engine_last_running_time = time_now(); \
} while(0)

#define StopEngine(engine) \
do { \
	engine_last_running_time = time_now(); \
	DestroyEngine(engine); \
	engine = NULL; \
} while(0)

#define InitEngineRunningTime() do { engine_last_running_time = 0; } while(0)
#define EngineIsIdle() (time_now() - engine_last_running_time > ENGINE_IDLE_TIME)

#endif // _ENGINE_H
