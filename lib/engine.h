/************************************************************************************
***
***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2021-01-13 21:19:01
***
************************************************************************************/

#ifndef _ENGINE_H
#define _ENGINE_H

#include <nimage/image.h>
#include <nimage/nnmsg.h>
#include <onnxruntime_c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "vision_service.h"

#define MAX_INPUT_TENSORS 8

// ONNX Runtime Engine
typedef struct {
  DWORD magic;
  char *model_path;
  int use_gpu;
  OrtEnv *env;
  OrtSession *session;
  OrtSessionOptions *session_options;

  // input node dims
  size_t n_input_nodes;
  int64_t input_node_dims[MAX_INPUT_TENSORS][4];

  // DAG only one output
  int64_t output_node_dims[4];

  std::vector<const char *> input_node_names;
  std::vector<const char *> output_node_names;
} OrtEngine;

// typedef int (*CustomSevice)(int socket, int service_code, TENSOR
// *input_tensor);
typedef int (*CustomSevice)(int, int, TENSOR *);

#define CheckEngine(e)                                                         \
  do {                                                                         \
    if (!ValidEngine(e)) {                                                     \
      fprintf(stderr, "Bad OrtEngine.\n");                                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int IsRunning(char *endpoint);

OrtEngine *CreateEngine(char *model_path, int use_gpu);
OrtEngine *CreateEngineFromArray(void *model_data, size_t model_data_length,
                                 int use_gpu);

int ValidEngine(OrtEngine *engine);
TENSOR *SingleTensorForward(OrtEngine *engine, TENSOR *input);
TENSOR *MultipleTensorForward(OrtEngine *engine, size_t n, TENSOR *inputs[]);
void DumpEngine(OrtEngine *engine);
void DestroyEngine(OrtEngine *engine);

int OnnxService(char *endpoint, char *onnx_file, int service_code, int use_gpu,
                CustomSevice custom_service_function);
int OnnxServiceFromArray(char *endpoint, void *model_data,
                         size_t model_data_length, int service_code,
                         int use_gpu, CustomSevice custom_service_function);

TENSOR *OnnxRPC(int socket, TENSOR *input, int reqcode);
TENSOR *ResizeOnnxRPC(int socket, TENSOR *send_tensor, int reqcode,
                      int multiples);
TENSOR *ZeropadOnnxRPC(int socket, TENSOR *send_tensor, int reqcode,
                       int multiples);

void SaveOutputImage(IMAGE *image, char *filename);
void SaveTensorAsImage(TENSOR *tensor, char *filename);
int CudaAvailable();

#define ENGINE_IDLE_TIME (120 * 1000) // 120 seconds
static TIME engine_last_running_time = 0;

#define StartEngine(engine, onnx_file, use_gpu)                                \
  do {                                                                         \
    if (engine == NULL)                                                        \
      engine = CreateEngine(onnx_file, use_gpu);                               \
    CheckEngine(engine);                                                       \
    engine_last_running_time = time_now();                                     \
  } while (0)

#define StartEngineFromArray(engine, model_data, model_data_length, use_gpu)   \
  do {                                                                         \
    if (engine == NULL) {                                                      \
      engine = CreateEngineFromArray(model_data, model_data_length, use_gpu);  \
    }                                                                          \
    CheckEngine(engine);                                                       \
    engine_last_running_time = time_now();                                     \
  } while (0)

#define StopEngine(engine)                                                     \
  do {                                                                         \
    engine_last_running_time = time_now();                                     \
    DestroyEngine(engine);                                                     \
    engine = NULL;                                                             \
  } while (0)

#define InitEngineRunningTime()                                                \
  do {                                                                         \
    engine_last_running_time = 0;                                              \
  } while (0)
#define EngineIsIdle()                                                         \
  (time_now() - engine_last_running_time > ENGINE_IDLE_TIME)

#endif // _ENGINE_H
