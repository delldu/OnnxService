/************************************************************************************
***
***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2021-01-12 23:52:44
***
************************************************************************************/

// Reference:
// https://github.com/microsoft/onnxruntime/edit/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp

#include <assert.h>

// For mkdir
#include <sys/stat.h>
#include <sys/types.h>

// opt/onnxruntime-linux-x64-gpu-1.6.0/include/cuda_provider_factory.h
#include <cuda_provider_factory.h>

// flock ...
#include <errno.h>
#include <sys/file.h>

#include "dcnv2_forward.h"
#include "engine.h"
#include "grid_sample.h"

// ONNX Runtime Engine
#define MAKE_FOURCC(a, b, c, d)                                                \
  (((DWORD)(a) << 24) | ((DWORD)(b) << 16) | ((DWORD)(c) << 8) |               \
   ((DWORD)(d) << 0))
#define ENGINE_MAGIC MAKE_FOURCC('O', 'N', 'R', 'T')

char *FindModel(char *modelname);

extern void CheckStatus(OrtStatus *status);
extern OrtValue *CreateOrtTensor(TENSOR *tensor, int gpu);

extern OrtValue *SingleForward(OrtEngine *engine, OrtValue *input_tensor);
extern OrtValue *MultipleForward(OrtEngine *engine, int n,
                                 const OrtValue *const *input_tensors);

extern int ValidOrtTensor(OrtValue *tensor);
extern size_t OrtTensorDimensions(OrtValue *tensor, int64_t *dims);
extern float *OrtTensorValues(OrtValue *tensor);
extern void DestroyOrtTensor(OrtValue *tensor);

const OrtApi *onnx_runtime_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
const char *c_CustomOpDomain = "onnxservice";
GridSampleOp c_GridSampleOp;
DCNv2ForwardOp c_DCNv2ForwardOp;

int RegisterOurOps(OrtSessionOptions *options) {
  OrtCustomOpDomain *domain = nullptr;

  if (onnx_runtime_api->CreateCustomOpDomain(c_CustomOpDomain, &domain)) {
    syslog_error("CreateCustomOpDomain");
    return RET_ERROR;
  }

  if (onnx_runtime_api->CustomOpDomain_Add(domain, &c_GridSampleOp)) {
    syslog_error("CustomOpDomain_Add: GridSampleOp");
    return RET_ERROR;
  }

  if (onnx_runtime_api->CustomOpDomain_Add(domain, &c_DCNv2ForwardOp)) {
    syslog_error("CustomOpDomain_Add: DCNv2ForwardOp");
    return RET_ERROR;
  }

  return onnx_runtime_api->AddCustomOpDomain(options, domain) ? RET_ERROR
                                                              : RET_OK;
}

int IsRunning(char *endpoint) {
  int i, n, fd, rc;
  char filename[256];
  char tempstr[256];

  n = strlen(endpoint);
  for (i = 0; i < n && i < (int)sizeof(tempstr) - 1; i++) {
    if (endpoint[i] == '/' || endpoint[i] == ':')
      tempstr[i] = '_';
    else
      tempstr[i] = endpoint[i];
  }
  tempstr[i] = '\0';

  snprintf(filename, sizeof(filename), "/tmp/%s.lock", tempstr);

  fd = open(filename, O_CREAT | O_RDWR, 0666);
  if ((rc = flock(fd, LOCK_EX | LOCK_NB))) {
    rc = (EWOULDBLOCK == errno) ? 1 : 0;
  } else {
    rc = 0;
  }
  if (rc)
    syslog_error("Service is running on %s ...", endpoint);

  return 0;
}

void InitInputNodes(OrtEngine *t) {
  size_t num_nodes;
  size_t num_dims;
  OrtAllocator *allocator;
  CheckStatus(onnx_runtime_api->GetAllocatorWithDefaultOptions(&allocator));

  CheckStatus(onnx_runtime_api->SessionGetInputCount(t->session, &num_nodes));

  t->n_input_nodes = num_nodes;
  syslog_info("Input nodes:");
  for (size_t i = 0; i < num_nodes && i < MAX_INPUT_TENSORS; i++) {
    char *name;

    CheckStatus(
        onnx_runtime_api->SessionGetInputName(t->session, i, allocator, &name));
    t->input_node_names.push_back(name);

    OrtTypeInfo *typeinfo;
    CheckStatus(
        onnx_runtime_api->SessionGetInputTypeInfo(t->session, i, &typeinfo));

    const OrtTensorTypeAndShapeInfo *tensor_info;
    CheckStatus(
        onnx_runtime_api->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

    ONNXTensorElementDataType type;
    CheckStatus(onnx_runtime_api->GetTensorElementType(tensor_info, &type));

    CheckStatus(onnx_runtime_api->GetDimensionsCount(tensor_info, &num_dims));
    if (num_dims > sizeof(t->input_node_dims[i]))
      num_dims = sizeof(t->input_node_dims[i]);

    printf("    no=%zu name=\"%s\" type=%d dims=%zu: ", i, name, type,
           num_dims);

    CheckStatus(onnx_runtime_api->GetDimensions(
        tensor_info, (int64_t *)t->input_node_dims[i], num_dims));

    for (size_t j = 0; j < num_dims; j++) {
      if (j < num_dims - 1)
        printf("%d x ", (int)t->input_node_dims[i][j]);
      else
        printf("%d\n", (int)t->input_node_dims[i][j]);
    }

    onnx_runtime_api->ReleaseTypeInfo(typeinfo);
  }
  // onnx_runtime_api->ReleaseAllocator(allocator); segmant fault !!!
}

void InitOutputNodes(OrtEngine *t) {
  size_t num_dims;
  OrtAllocator *allocator;

  CheckStatus(onnx_runtime_api->GetAllocatorWithDefaultOptions(&allocator));

  size_t num_nodes;
  CheckStatus(onnx_runtime_api->SessionGetOutputCount(t->session, &num_nodes));

  syslog_info("Output nodes:");
  for (size_t i = 0; i < num_nodes; i++) {
    char *name;

    CheckStatus(onnx_runtime_api->SessionGetOutputName(t->session, i, allocator,
                                                       &name));
    t->output_node_names.push_back(name);

    OrtTypeInfo *typeinfo;
    CheckStatus(
        onnx_runtime_api->SessionGetOutputTypeInfo(t->session, i, &typeinfo));

    const OrtTensorTypeAndShapeInfo *tensor_info;
    CheckStatus(
        onnx_runtime_api->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

    ONNXTensorElementDataType type;
    CheckStatus(onnx_runtime_api->GetTensorElementType(tensor_info, &type));

    CheckStatus(onnx_runtime_api->GetDimensionsCount(tensor_info, &num_dims));
    if (num_dims > sizeof(t->output_node_dims))
      num_dims = sizeof(t->output_node_dims);
    printf("    no=%zu name=\"%s\" type=%d dims=%zu: ", i, name, type,
           num_dims);

    CheckStatus(onnx_runtime_api->GetDimensions(
        tensor_info, (int64_t *)t->output_node_dims, num_dims));
    for (size_t j = 0; j < num_dims && j < sizeof(t->output_node_dims); j++) {
      if (j < num_dims - 1)
        printf("%d x ", (int)t->output_node_dims[j]);
      else
        printf("%d\n", (int)t->output_node_dims[j]);
    }

    onnx_runtime_api->ReleaseTypeInfo(typeinfo);
  }
  // onnx_runtime_api->ReleaseAllocator(allocator); segment fault !!!
}

void CheckStatus(OrtStatus *status) {
  if (status != NULL) {
    const char *msg = onnx_runtime_api->GetErrorMessage(status);
    syslog_error("%s\n", msg);
    onnx_runtime_api->ReleaseStatus(status);
    exit(1);
  }
}

int ValidOrtTensor(OrtValue *tensor) {
  int is_tensor;
  CheckStatus(onnx_runtime_api->IsTensor(tensor, &is_tensor));

  if (!is_tensor) {
    syslog_error("Tensor is not valid\n");
  }
  return is_tensor;
}

OrtValue *CreateOrtTensor(TENSOR *tensor, int gpu) {
  size_t size, n_dims;
  int64_t dims[4];
  OrtStatus *status;
  OrtValue *orttensor = NULL;

  n_dims = 4;
  dims[0] = tensor->batch;
  dims[1] = tensor->chan;
  dims[2] = tensor->height;
  dims[3] = tensor->width;
  size = tensor->batch * tensor->chan * tensor->height * tensor->width;

  OrtMemoryInfo *memory_info;
  if (gpu) {
    CheckStatus(onnx_runtime_api->CreateMemoryInfo(
        "Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &memory_info));
  } else {
    CheckStatus(onnx_runtime_api->CreateMemoryInfo(
        "Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault, &memory_info));
  }

  status = onnx_runtime_api->CreateTensorWithDataAsOrtValue(
      memory_info, tensor->data, size * sizeof(float), dims, n_dims,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &orttensor);
  CheckStatus(status);
  onnx_runtime_api->ReleaseMemoryInfo(memory_info);

  ValidOrtTensor(orttensor);

  return orttensor;
}

size_t OrtTensorDimensions(OrtValue *tensor, int64_t *dims) {
  size_t dim_count;

  struct OrtTensorTypeAndShapeInfo *shape_info;
  CheckStatus(onnx_runtime_api->GetTensorTypeAndShape(tensor, &shape_info));

  CheckStatus(onnx_runtime_api->GetDimensionsCount(shape_info, &dim_count));
  if (dim_count < 1) {
    syslog_error("Tensor must have 4 dimensions");
    exit(-1);
  }
  if (dim_count > 4)
    dim_count = 4; // Truncate for BxCxHxW format

  CheckStatus(onnx_runtime_api->GetDimensions(shape_info, dims, dim_count));

  onnx_runtime_api->ReleaseTensorTypeAndShapeInfo(shape_info);

  return dim_count;
}

float *OrtTensorValues(OrtValue *tensor) {
  float *floatarray;
  CheckStatus(
      onnx_runtime_api->GetTensorMutableData(tensor, (void **)&floatarray));
  return floatarray;
}

void DestroyOrtTensor(OrtValue *tensor) {
  onnx_runtime_api->ReleaseValue(tensor);
}

OrtEngine *CreateEngine(char *model_path, int use_gpu) {
  OrtEngine *t;

  syslog_info("Creating ONNX Runtime Engine for model %s ...", model_path);

  t = (OrtEngine *)calloc((size_t)1, sizeof(OrtEngine));
  if (!t) {
    syslog_error("Allocate memeory.");
    return NULL;
  }
  t->magic = ENGINE_MAGIC;
  t->model_path = FindModel(model_path);
  t->use_gpu = use_gpu;

  // Building ...
  CheckStatus(onnx_runtime_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                          "OrtEngine", &(t->env)));
  // CheckStatus(onnx_runtime_api->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE,
  // "OrtEngine", &(t->env)));

  // initialize session options if needed
  CheckStatus(onnx_runtime_api->CreateSessionOptions(&(t->session_options)));
  // CheckStatus(onnx_runtime_api->SetIntraOpNumThreads(t->session_options, 0));
  // // 0 -- for default

  // RegisterOurOps, support onnx::grid_sampler
  RegisterOurOps(t->session_options);

  // Sets graph optimization level
  CheckStatus(onnx_runtime_api->SetSessionGraphOptimizationLevel(
      t->session_options, ORT_ENABLE_ALL));
  // ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL

  // Optionally add more execution providers via session_options
  // E.g. for CUDA include cuda_provider_factory.h and uncomment the following
  // line:

  if (use_gpu)
    CheckStatus(
        OrtSessionOptionsAppendExecutionProvider_CUDA(t->session_options, 0));

  // CreateSessionFromArray prototype
  // ORT_API2_STATUS(CreateSessionFromArray, _In_ const OrtEnv* env,
  //      _In_ const void* model_data, size_t model_data_length,
  //      _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);
  CheckStatus(onnx_runtime_api->CreateSession(
      t->env, model_path, t->session_options, &(t->session)));

  // Setup input_node_names;
  InitInputNodes(t);

  // Setup output_node_names;
  InitOutputNodes(t);

  syslog_info("Create ONNX Runtime Engine OK.\n");

  return t;
}

OrtEngine *CreateEngineFromArray(void *model_data, size_t model_data_length,
                                 int use_gpu) {
  OrtEngine *t;
  char *model_path = strdup("memory");

  syslog_info("Creating ONNX Runtime Engine for model size %s ...", model_path);

  t = (OrtEngine *)calloc((size_t)1, sizeof(OrtEngine));
  if (!t) {
    syslog_error("Allocate memeory.");
    return NULL;
  }
  t->magic = ENGINE_MAGIC;
  t->model_path = model_path;
  t->use_gpu = use_gpu;

  // Building ...
  CheckStatus(onnx_runtime_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                          "OrtEngine", &(t->env)));
  // CheckStatus(onnx_runtime_api->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE,
  // "OrtEngine", &(t->env)));

  // initialize session options if needed
  CheckStatus(onnx_runtime_api->CreateSessionOptions(&(t->session_options)));
  // CheckStatus(onnx_runtime_api->SetIntraOpNumThreads(t->session_options, 0));
  // // 0 -- for default

  // RegisterOurOps, support onnx::grid_sampler
  RegisterOurOps(t->session_options);

  // Sets graph optimization level
  CheckStatus(onnx_runtime_api->SetSessionGraphOptimizationLevel(
      t->session_options, ORT_ENABLE_ALL));
  // ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL

  // Optionally add more execution providers via session_options
  // E.g. for CUDA include cuda_provider_factory.h and uncomment the following
  // line:

  if (use_gpu)
    CheckStatus(
        OrtSessionOptionsAppendExecutionProvider_CUDA(t->session_options, 0));

  // CreateSessionFromArray prototype
  // ORT_API2_STATUS(CreateSessionFromArray, _In_ const OrtEnv* env,
  //      _In_ const void* model_data, size_t model_data_length,
  //      _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);
  CheckStatus(onnx_runtime_api->CreateSessionFromArray(
      t->env, model_data, model_data_length, t->session_options,
      &(t->session)));

  // Setup input_node_names;
  InitInputNodes(t);

  // Setup output_node_names;
  InitOutputNodes(t);

  syslog_info("Create ONNX Runtime Engine OK.\n");

  return t;
}

int ValidEngine(OrtEngine *t) {
  return (!t || t->magic != ENGINE_MAGIC) ? 0 : 1;
}

OrtValue *SingleForward(OrtEngine *engine, OrtValue *input_tensor) {
  OrtStatus *status;
  OrtValue *output_tensor = NULL;

  ValidOrtTensor(input_tensor);

  /* prototype
     ORT_API2_STATUS(Run, _Inout_ OrtSession* sess, _In_opt_ const
     OrtRunOptions* run_options, _In_reads_(input_len) const char* const*
     input_names, _In_reads_(input_len) const OrtValue* const* input, size_t
     input_len, _In_reads_(output_names_len) const char* const* output_names1,
     size_t output_names_len, _Inout_updates_all_(output_names_len) OrtValue**
     output);
   */
  status = onnx_runtime_api->Run(
      engine->session, NULL, engine->input_node_names.data(),
      (const OrtValue *const *)&input_tensor, 1,
      engine->output_node_names.data(), 1, &output_tensor);

  CheckStatus(status);

  ValidOrtTensor(output_tensor);
  return output_tensor;
}

TENSOR *SingleTensorForward(OrtEngine *engine, TENSOR *input) {
  size_t i, size, n_dims;
  int64_t dims[4];
  OrtValue *input_ortvalue, *output_ortvalue;
  TENSOR *temp_tensor;
  TENSOR *output = NULL;

  CHECK_TENSOR(input);

  // Format input ...
  dims[0] = input->batch;
  dims[1] = input->chan;
  dims[2] = input->height;
  dims[3] = input->width;
  for (i = 0; i < 4; i++) {
    if (engine->input_node_dims[0][i] > 0)
      dims[i] = (int)engine->input_node_dims[0][i];
  }
  temp_tensor = tensor_reshape(input, dims[0], dims[1], dims[2], dims[3]);
  CHECK_TENSOR(temp_tensor);
  input_ortvalue = CreateOrtTensor(temp_tensor, engine->use_gpu);

  output_ortvalue = SingleForward(engine, input_ortvalue);
  // Bug Fix: temp_tensor share data with input_ortvalue, must destroy after
  // SingleForward  ...
  tensor_destroy(temp_tensor);

  // Format output ...
  if (ValidOrtTensor(output_ortvalue)) {
    n_dims = OrtTensorDimensions(output_ortvalue, dims);
    if (n_dims > 0) {
      if (n_dims < 4) {
        // Format: BxCxHxW
        for (i = 0; i < n_dims; i++)
          dims[3 - i] = dims[n_dims - i - 1];
        for (i = 0; i < 4 - n_dims; i++)
          dims[i] = 1;
      }
      output = tensor_create(dims[0], dims[1], dims[2], dims[3]);
      CHECK_TENSOR(output);
      size = output->batch * output->chan * output->height * output->width;
      memcpy(output->data, OrtTensorValues(output_ortvalue),
             size * sizeof(float));
    }
    DestroyOrtTensor(output_ortvalue);
  }

  DestroyOrtTensor(input_ortvalue);

  return output;
}

OrtValue *MultipleForward(OrtEngine *engine, int n,
                          const OrtValue *const *input_tensors) {
  OrtStatus *status;
  OrtValue *output_tensor = NULL;

  /* prototype
     ORT_API2_STATUS(Run, _Inout_ OrtSession* sess, _In_opt_ const
     OrtRunOptions* run_options, _In_reads_(input_len) const char* const*
     input_names, _In_reads_(input_len) const OrtValue* const* input, size_t
     input_len, _In_reads_(output_names_len) const char* const* output_names1,
     size_t output_names_len, _Inout_updates_all_(output_names_len) OrtValue**
     output);
   */
  status = onnx_runtime_api->Run(
      engine->session, NULL, engine->input_node_names.data(), input_tensors, n,
      engine->output_node_names.data(), 1, &output_tensor);

  CheckStatus(status);

  ValidOrtTensor(output_tensor);
  return output_tensor;
}

TENSOR *MultipleTensorForward(OrtEngine *engine, size_t n, TENSOR *inputs[]) {
  int64_t dims[4];
  size_t i, j, size, n_dims;
  OrtValue *input_ortvalues[MAX_INPUT_TENSORS], *output_ortvalue;
  TENSOR *temp_tensors[MAX_INPUT_TENSORS];
  TENSOR *output = NULL;

  if (n != engine->n_input_nodes) {
    syslog_error("Engine expected %d input tensor, but got %d.",
                 engine->n_input_nodes, n);
    return NULL;
  }
  for (i = 0; i < n; i++)
    CHECK_TENSOR(inputs[i]);

  // n == engine->n_input_nodes !!!
  // Format inputs ...
  for (i = 0; i < n; i++) {
    dims[0] = inputs[i]->batch;
    dims[1] = inputs[i]->chan;
    dims[2] = inputs[i]->height;
    dims[3] = inputs[i]->width;
    for (j = 0; j < 4; j++) {
      if (engine->input_node_dims[i][j] > 0)
        dims[j] = (int)engine->input_node_dims[i][j];
    }
    temp_tensors[i] =
        tensor_reshape(inputs[i], dims[0], dims[1], dims[2], dims[3]);
    CHECK_TENSOR(temp_tensors[i]);

    input_ortvalues[i] = CreateOrtTensor(temp_tensors[i], engine->use_gpu);
  }

  output_ortvalue = MultipleForward(engine, n, input_ortvalues);

  for (i = 0; i < n; i++) {
    // Bug Fix: temp_tensors[i] share data with input_ortvalues[i], so destroy
    // after forward ...
    tensor_destroy(temp_tensors[i]);
  }

  // Format output ...
  if (ValidOrtTensor(output_ortvalue)) {
    n_dims = OrtTensorDimensions(output_ortvalue, dims);
    if (n_dims > 0) {
      if (n_dims < 4) {
        // Format: BxCxHxW
        for (i = 0; i < n_dims; i++)
          dims[3 - i] = dims[n_dims - i - 1];
        for (i = 0; i < 4 - n_dims; i++)
          dims[i] = 1;
      }
      output = tensor_create(dims[0], dims[1], dims[2], dims[3]);
      CHECK_TENSOR(output);
      size = output->batch * output->chan * output->height * output->width;
      memcpy(output->data, OrtTensorValues(output_ortvalue),
             size * sizeof(float));
    }
    DestroyOrtTensor(output_ortvalue);
  }

  for (i = 0; i < n; i++) {
    DestroyOrtTensor(input_ortvalues[i]);
  }

  return output;
}

void DumpEngine(OrtEngine *engine) {
  size_t i;
  char buf[256];

  if (ValidEngine(engine)) {
    syslog_info("Engine:");

    for (i = 0; i < engine->n_input_nodes; i++) {
      snprintf(buf, sizeof(buf) - 1, "    Input %s: %d x %d %d x %d",
               engine->input_node_names[i], (int)engine->input_node_dims[i][0],
               (int)engine->input_node_dims[i][1],
               (int)engine->input_node_dims[i][2],
               (int)engine->input_node_dims[i][3]);
      syslog_info("%s", buf);
    }

    snprintf(buf, sizeof(buf) - 1, "    Output %s: %d x %d %d x %d",
             engine->output_node_names[0], (int)engine->output_node_dims[0],
             (int)engine->output_node_dims[1], (int)engine->output_node_dims[2],
             (int)engine->output_node_dims[3]);
    syslog_info("%s", buf);
  } else {
    syslog_info("Engine == NULL");
  }
}

void DestroyEngine(OrtEngine *engine) {
  if (!ValidEngine(engine))
    return;

  if (engine->model_path)
    free(engine->model_path);

  // Release ...
  engine->input_node_names.clear();
  engine->output_node_names.clear();

  onnx_runtime_api->ReleaseSession(engine->session);
  onnx_runtime_api->ReleaseSessionOptions(engine->session_options);
  onnx_runtime_api->ReleaseEnv(engine->env);

  free(engine);
}

int OnnxService(char *endpoint, char *onnx_file, int service_code, int use_gpu,
                CustomSevice custom_service_function) {
  int socket, count, msgcode;
  TENSOR *input_tensor, *output_tensor;
  OrtEngine *engine = NULL;

  if ((socket = server_open(endpoint)) < 0)
    return RET_ERROR;

  if (!custom_service_function)
    custom_service_function = service_response;

  count = 0;
  for (;;) {
    if (EngineIsIdle())
      StopEngine(engine);

    if (!socket_readable(socket, 1000)) // timeout 1 s
      continue;

    input_tensor = service_request(socket, &msgcode);
    if (!tensor_valid(input_tensor))
      continue;

    if (msgcode == service_code) {
      syslog_info("Service %d times", count);
      StartEngine(engine, onnx_file, use_gpu);

      // Real service ...
      time_reset();
      output_tensor = SingleTensorForward(engine, input_tensor);
      time_spend((char *)"Predict");

      service_response(socket, service_code, output_tensor);
      tensor_destroy(output_tensor);

      count++;
    } else {
      custom_service_function(socket, msgcode, input_tensor);
    }

    tensor_destroy(input_tensor);
  }
  StopEngine(engine);

  syslog(LOG_INFO, "Service shutdown.\n");
  server_close(socket);

  return RET_OK;
}

int OnnxServiceFromArray(char *endpoint, void *model_data,
                         size_t model_data_length, int service_code,
                         int use_gpu, CustomSevice custom_service_function) {
  int socket, count, msgcode;
  TENSOR *input_tensor, *output_tensor;
  OrtEngine *engine;

  if ((socket = server_open(endpoint)) < 0)
    return RET_ERROR;

  if (!custom_service_function)
    custom_service_function = service_response;

  count = 0;
  for (;;) {
    if (EngineIsIdle())
      StopEngine(engine);

    if (!socket_readable(socket, 1000)) // timeout 1 s
      continue;

    input_tensor = service_request(socket, &msgcode);
    if (!tensor_valid(input_tensor))
      continue;

    if (msgcode == service_code) {
      syslog_info("Service %d times", count);
      StartEngineFromArray(engine, model_data, model_data_length, use_gpu);

      // Real service ...
      time_reset();
      output_tensor = SingleTensorForward(engine, input_tensor);
      time_spend((char *)"Predict");
      service_response(socket, service_code, output_tensor);
      tensor_destroy(output_tensor);
      count++;
    } else {
      custom_service_function(socket, msgcode, input_tensor);
    }

    tensor_destroy(input_tensor);
  }
  StopEngine(engine);

  syslog(LOG_INFO, "Service shutdown.\n");
  server_close(socket);

  return RET_OK;
}

TENSOR *OnnxRPC(int socket, TENSOR *input, int reqcode) {
  int rescode = -1;
  TENSOR *output = NULL;

  CHECK_TENSOR(input);

  if (tensor_send(socket, reqcode, input) == RET_OK) {
    output = tensor_recv(socket, &rescode);
  }
  if (rescode != reqcode) {
    // Bad service response
    syslog_error("Remote running service.");
    tensor_destroy(output);
    return NULL;
  }
  return output;
}

TENSOR *ResizeOnnxRPC(int socket, TENSOR *send_tensor, int reqcode,
                      int multiples) {
  int nh, nw;
  TENSOR *resize_send, *resize_recv, *recv_tensor;

  CHECK_TENSOR(send_tensor);

  nh = (send_tensor->height + multiples - 1) / multiples;
  nh *= multiples;
  nw = (send_tensor->width + multiples - 1) / multiples;
  nw *= multiples;

  if (send_tensor->height == nh && send_tensor->width == nw) {
    // Normal onnx RPC
    recv_tensor = OnnxRPC(socket, send_tensor, reqcode);
  } else {
    resize_send = tensor_zoom(send_tensor, nh, nw);
    CHECK_TENSOR(resize_send);
    resize_recv = OnnxRPC(socket, resize_send, reqcode);
    recv_tensor =
        tensor_zoom(resize_recv, send_tensor->height, send_tensor->width);
    tensor_destroy(resize_recv);
    tensor_destroy(resize_send);
  }

  return recv_tensor;
}

TENSOR *ZeropadOnnxRPC(int socket, TENSOR *send_tensor, int reqcode,
                       int multiples) {
  int nh, nw;
  TENSOR *resize_send, *resize_recv, *recv_tensor;

  CHECK_TENSOR(send_tensor);

  nh = (send_tensor->height + multiples - 1) / multiples;
  nh *= multiples;
  nw = (send_tensor->width + multiples - 1) / multiples;
  nw *= multiples;

  if (send_tensor->height == nh && send_tensor->width == nw) {
    // Normal onnx RPC
    recv_tensor = OnnxRPC(socket, send_tensor, reqcode);
  } else {
    resize_send = tensor_zeropad(send_tensor, nh, nw);
    CHECK_TENSOR(resize_send);
    resize_recv = OnnxRPC(socket, resize_send, reqcode);
    recv_tensor =
        tensor_zeropad(resize_recv, send_tensor->height, send_tensor->width);
    tensor_destroy(resize_recv);
    tensor_destroy(resize_send);
  }

  return recv_tensor;
}

void SaveOutputImage(IMAGE *image, char *filename) {
  char output_filename[256], *p;

  mkdir("output", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  if (image_valid(image)) {
    p = strrchr(filename, '/');
    p = (!p) ? filename : p + 1;
    snprintf(output_filename, sizeof(output_filename) - 1, "output/%s", p);
    image_save(image, output_filename);
  }
}

void SaveTensorAsImage(TENSOR *tensor, char *filename) {
  IMAGE *image = image_from_tensor(tensor, 0);

  if (image_valid(image)) {
    SaveOutputImage(image, filename);
    image_destroy(image);
  }
}

int CudaAvailable() {
  int ok = 0;
  OrtStatus *status;
  OrtSessionOptions *session_options;

  CheckStatus(onnx_runtime_api->CreateSessionOptions(&session_options));
  status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
  if (status != NULL) {
    const char *msg = onnx_runtime_api->GetErrorMessage(status);
    syslog_error("%s", msg);
    onnx_runtime_api->ReleaseStatus(status);
  } else {
    ok = 1;
  }
  onnx_runtime_api->ReleaseSessionOptions(session_options);
  return ok;
}

char *FindModel(char *modelname) {
  char filename[256];

  snprintf(filename, sizeof(filename), "%s", modelname);
  if (access(filename, F_OK) == 0) {
    CheckPoint("Found Model: %s", filename);
    return strdup(filename);
  }

  snprintf(filename, sizeof(filename), "%s/%s", ONNXMODEL_INSTALL_DIR,
           modelname);
  if (access(filename, F_OK) == 0) {
    CheckPoint("Found Model: %s", filename);
    return strdup(filename);
  }

  syslog_error("Model %s NOT Found !", modelname);
  return NULL;
}
