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

OrtEngine *CreateEngine(char *model_path, int use_gpu);
OrtEngine *CreateEngineFromArray(void* model_data, size_t model_data_length, int use_gpu);

int ValidEngine(OrtEngine * engine);
TENSOR *TensorForward(OrtEngine * engine, TENSOR * input);
void DestroyEngine(OrtEngine * engine);

int OnnxService(char *endpoint, char *onnx_file, int service_code, int use_gpu);
int OnnxServiceFromArray(char *endpoint, void* model_data, size_t model_data_length, int service_code, int use_gpu);

TENSOR *OnnxRPC(int socket, TENSOR * input, int reqcode);
TENSOR *ResizeOnnxRPC(int socket, TENSOR *send_tensor, int reqcode, int multiples);
TENSOR *ZeropadOnnxRPC(int socket, TENSOR *send_tensor, int reqcode, int multiples);

void SaveOutputImage(IMAGE *image, char *filename);
void SaveTensorAsImage(TENSOR *tensor, char *filename);


#define ENGINE_IDLE_TIME (120*1000)	// 120 seconds
static TIME engine_last_running_time = 0;

#define StartEngine(engine, onnx_file, use_gpu) \
do { \
	if (engine == NULL) \
		engine = CreateEngine(onnx_file, use_gpu); \
	CheckEngine(engine); \
} while(0)

#define StartEngineFromArray(engine, model_data, model_data_length, use_gpu) \
do { \
	if (engine == NULL) { \
		syslog_info("---- Start Engine ..."); \
		engine = CreateEngineFromArray(model_data, model_data_length, use_gpu); \
	} \
	CheckEngine(engine); \
} while(0)

#define StopEngine(engine) \
do { \
	syslog_info("---- Stop Engine ..."); \
	DestroyEngine(engine); \
	engine = NULL; \
	engine_last_running_time = time_now(); \
} while(0)

#define InitEngineRunningTime() do { engine_last_running_time = 0; } while(0)
#define EngineIsIdle() (time_now() - engine_last_running_time > ENGINE_IDLE_TIME)
#define UpdateEngineRunningTime() do { engine_last_running_time = time_now(); } while(0)

#endif // _ENGINE_H
