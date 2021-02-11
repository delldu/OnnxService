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
#include "engine.h"

// opt/onnxruntime-linux-x64-gpu-1.6.0/include/cuda_provider_factory.h
#include <cuda_provider_factory.h>

// ONNX Runtime Engine
#define MAKE_FOURCC(a,b,c,d) (((DWORD)(a) << 24) | ((DWORD)(b) << 16) | ((DWORD)(c) << 8) | ((DWORD)(d) << 0))
#define ENGINE_MAGIC MAKE_FOURCC('O', 'N', 'R', 'T')
const OrtApi *onnx_runtime_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);


void InitInputNodes(OrtEngine * t)
{
	size_t num_nodes;
	size_t num_dims;
	int64_t input_node_dims[4];
	OrtAllocator *allocator;
	CheckStatus(onnx_runtime_api->GetAllocatorWithDefaultOptions(&allocator));

	CheckStatus(onnx_runtime_api->SessionGetInputCount(t->session, &num_nodes));

	syslog_info("Input nodes:");
	for (size_t i = 0; i < num_nodes; i++) {
		char *name;

		CheckStatus(onnx_runtime_api->SessionGetInputName(t->session, i, allocator, &name));
		t->input_node_names.push_back(name);

		OrtTypeInfo *typeinfo;
		CheckStatus(onnx_runtime_api->SessionGetInputTypeInfo(t->session, i, &typeinfo));

		const OrtTensorTypeAndShapeInfo *tensor_info;
		CheckStatus(onnx_runtime_api->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

		ONNXTensorElementDataType type;
		CheckStatus(onnx_runtime_api->GetTensorElementType(tensor_info, &type));

		CheckStatus(onnx_runtime_api->GetDimensionsCount(tensor_info, &num_dims));
		if (num_dims > 4)
			num_dims = 4;

		printf("    no=%zu name=\"%s\" type=%d dims=%zu: ", i, name, type, num_dims);

		CheckStatus(onnx_runtime_api->GetDimensions(tensor_info, (int64_t *) input_node_dims, num_dims));

		for (size_t j = 0; j < num_dims; j++) {
			if (j < num_dims - 1)
				printf("%jd x ", input_node_dims[j]);	// xxxx8888
			else
				printf("%jd\n", input_node_dims[j]);
		}

		onnx_runtime_api->ReleaseTypeInfo(typeinfo);
	}
	// onnx_runtime_api->ReleaseAllocator(allocator); segmant fault !!!
}

void InitOutputNodes(OrtEngine * t)
{
	size_t num_dims;
	int64_t output_node_dims[4];

	OrtAllocator *allocator;

	CheckStatus(onnx_runtime_api->GetAllocatorWithDefaultOptions(&allocator));

	size_t num_nodes;
	CheckStatus(onnx_runtime_api->SessionGetOutputCount(t->session, &num_nodes));

	syslog_info("Output nodes:");
	for (size_t i = 0; i < num_nodes; i++) {
		char *name;

		CheckStatus(onnx_runtime_api->SessionGetOutputName(t->session, i, allocator, &name));
		t->output_node_names.push_back(name);

		OrtTypeInfo *typeinfo;
		CheckStatus(onnx_runtime_api->SessionGetOutputTypeInfo(t->session, i, &typeinfo));

		const OrtTensorTypeAndShapeInfo *tensor_info;
		CheckStatus(onnx_runtime_api->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

		ONNXTensorElementDataType type;
		CheckStatus(onnx_runtime_api->GetTensorElementType(tensor_info, &type));

		CheckStatus(onnx_runtime_api->GetDimensionsCount(tensor_info, &num_dims));
		if (num_dims > 4)
			num_dims = 4;
		printf("    no=%zu name=\"%s\" type=%d dims=%zu: ", i, name, type, num_dims);

		CheckStatus(onnx_runtime_api->GetDimensions(tensor_info, (int64_t *) output_node_dims, num_dims));
		for (size_t j = 0; j < num_dims; j++) {
			if (j < num_dims - 1)
				printf("%jd x ", output_node_dims[j]);
			else
				printf("%jd\n", output_node_dims[j]);
		}

		onnx_runtime_api->ReleaseTypeInfo(typeinfo);
	}
	// onnx_runtime_api->ReleaseAllocator(allocator); segmant fault !!!
}

void CheckStatus(OrtStatus * status)
{
	if (status != NULL) {
		const char *msg = onnx_runtime_api->GetErrorMessage(status);
		syslog_error("%s\n", msg);
		onnx_runtime_api->ReleaseStatus(status);
		exit(1);
	}
}

int ValidOrtTensor(OrtValue * tensor)
{
	int is_tensor;
	CheckStatus(onnx_runtime_api->IsTensor(tensor, &is_tensor));

	if (!is_tensor) {
		syslog_error("Tensor is not valid\n");
	}
	return is_tensor;
}

#if 0
OrtValue *CreateOrtTensor(TENSOR * tensor)
{
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
	CheckStatus(onnx_runtime_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
	status = onnx_runtime_api->CreateTensorWithDataAsOrtValue(memory_info,
															  tensor->data, size * sizeof(float),
															  dims, n_dims,
															  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &orttensor);
	CheckStatus(status);
	onnx_runtime_api->ReleaseMemoryInfo(memory_info);

	ValidOrtTensor(orttensor);

	return orttensor;
}
#endif

OrtValue* CreateOrtTensor(TENSOR * tensor, int gpu)
{
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
		CheckStatus(onnx_runtime_api->CreateMemoryInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &memory_info));
	} else {
		CheckStatus(onnx_runtime_api->CreateMemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault, &memory_info));
	}

	status = onnx_runtime_api->CreateTensorWithDataAsOrtValue(memory_info,
															  tensor->data, size * sizeof(float),
															  dims, n_dims,
															  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &orttensor);
	CheckStatus(status);
	onnx_runtime_api->ReleaseMemoryInfo(memory_info);

	ValidOrtTensor(orttensor);

	return orttensor;
}

size_t OrtTensorDimensions(OrtValue * tensor, int64_t * dims)
{
	size_t dim_count;

	struct OrtTensorTypeAndShapeInfo *shape_info;
	CheckStatus(onnx_runtime_api->GetTensorTypeAndShape(tensor, &shape_info));

	CheckStatus(onnx_runtime_api->GetDimensionsCount(shape_info, &dim_count));
	if (dim_count < 1) {
		syslog_error("Tensor must have 4 dimensions");
		exit(-1);
	}
	if (dim_count > 4)
		dim_count = 4;	// Truncate for BxCxHxW format

	CheckStatus(onnx_runtime_api->GetDimensions(shape_info, dims, dim_count));

	onnx_runtime_api->ReleaseTensorTypeAndShapeInfo(shape_info);

	return dim_count;
}

float *OrtTensorValues(OrtValue * tensor)
{
	float *floatarray;
	CheckStatus(onnx_runtime_api->GetTensorMutableData(tensor, (void **) &floatarray));
	return floatarray;
}

void DestroyOrtTensor(OrtValue * tensor)
{
	onnx_runtime_api->ReleaseValue(tensor);
}

OrtEngine *CreateEngine(const char *model_path, int use_gpu)
{
	OrtEngine *t;

	syslog_info("Creating ONNX Runtime Engine for model %s ...\n", model_path);

	t = (OrtEngine *) calloc((size_t) 1, sizeof(OrtEngine));
	if (!t) {
		syslog_error("Allocate memeory.");
		return NULL;
	}
	t->magic = ENGINE_MAGIC;
	t->model_path = model_path;
	t->use_gpu = use_gpu;

	// Building ...
	CheckStatus(onnx_runtime_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "OrtEngine", &(t->env)));
	// CheckStatus(onnx_runtime_api->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "OrtEngine", &(t->env)));

	// initialize session options if needed
	CheckStatus(onnx_runtime_api->CreateSessionOptions(&(t->session_options)));
	// CheckStatus(onnx_runtime_api->SetIntraOpNumThreads(t->session_options, 0));  // 0 -- for default 

	// Sets graph optimization level
	CheckStatus(onnx_runtime_api->SetSessionGraphOptimizationLevel(t->session_options, ORT_ENABLE_ALL));
	// ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL

	// Optionally add more execution providers via session_options
	// E.g. for CUDA include cuda_provider_factory.h and uncomment the following line:
	if (use_gpu)
		CheckStatus(OrtSessionOptionsAppendExecutionProvider_CUDA(t->session_options, 0));

	CheckStatus(onnx_runtime_api->CreateSession(t->env, model_path, t->session_options, &(t->session)));

	// Setup input_node_names;
	InitInputNodes(t);

	// Setup output_node_names;
	InitOutputNodes(t);

	syslog_info("Create ONNX Runtime Engine OK.\n");

	return t;
}

int ValidEngine(OrtEngine * t)
{
	return (!t || t->magic != ENGINE_MAGIC) ? 0 : 1;
}

OrtValue *SimpleForward(OrtEngine * engine, OrtValue * input_tensor)
{
	OrtStatus *status;
	OrtValue *output_tensor = NULL;

	ValidOrtTensor(input_tensor);

	/* prototype
	   ORT_API2_STATUS(Run, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
	   _In_reads_(input_len) const char* const* input_names,
	   _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
	   _In_reads_(output_names_len) const char* const* output_names1, size_t output_names_len,
	   _Inout_updates_all_(output_names_len) OrtValue** output);
	 */
	status = onnx_runtime_api->Run(engine->session, NULL,
								   engine->input_node_names.data(), (const OrtValue * const *) &input_tensor, 1,
								   engine->output_node_names.data(), 1, &output_tensor);

	CheckStatus(status);

	ValidOrtTensor(output_tensor);
	return output_tensor;
}

TENSOR *TensorForward(OrtEngine * engine, TENSOR * input)
{
	size_t i, size, n_dims;
	int64_t dims[4];
	OrtValue *input_ortvalue, *output_ortvalue;
	TENSOR *output = NULL;

	CHECK_TENSOR(input);

	input_ortvalue = CreateOrtTensor(input, engine->use_gpu);

	output_ortvalue = SimpleForward(engine, input_ortvalue);
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
			output = tensor_create((WORD) dims[0], (WORD) dims[1], (WORD) dims[2], (WORD) dims[3]);
			CHECK_TENSOR(output);
			size = output->batch * output->chan * output->height * output->width;
			memcpy(output->data, OrtTensorValues(output_ortvalue), size * sizeof(float));
		}
		DestroyOrtTensor(output_ortvalue);
	}

	DestroyOrtTensor(input_ortvalue);

	return output;
}

void DestroyEngine(OrtEngine * engine)
{
	if (!ValidEngine(engine))
		return;

	// Release ...
	engine->input_node_names.clear();
	engine->output_node_names.clear();

	onnx_runtime_api->ReleaseSession(engine->session);
	onnx_runtime_api->ReleaseSessionOptions(engine->session_options);
	onnx_runtime_api->ReleaseEnv(engine->env);

	free(engine);
}

int OnnxService(char *endpoint, char *onnx_file, int use_gpu)
{
	float option;
	int socket, reqcode, count, rescode;
	TENSOR *input_tensor, *output_tensor;
	OrtEngine *engine;

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	engine = CreateEngine(onnx_file, use_gpu);
	CheckEngine(engine);

	count = 0;
	for (;;) {
		syslog_info("Service %d times", count);

		input_tensor = request_recv(socket, &reqcode, &option);

		if (!tensor_valid(input_tensor)) {
			syslog_error("Request recv bad tensor ...");
			continue;
		}
		syslog_info("Request Code = %d, Option = %f", reqcode, option);

		// Real service ...
		time_reset();
		output_tensor = TensorForward(engine, input_tensor);
		time_spend((char *)"Infer");

		if (tensor_valid(output_tensor)) {
			rescode = reqcode;
			response_send(socket, output_tensor, rescode);
			tensor_destroy(output_tensor);
		}

		tensor_destroy(input_tensor);

		count++;
	}
	DestroyEngine(engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

TENSOR *OnnxRPC(int socket, TENSOR * input, int reqcode, float option, int *rescode)
{
	TENSOR *output = NULL;

	CHECK_TENSOR(input);

	if (request_send(socket, reqcode, input, option) == RET_OK) {
		output = response_recv(socket, rescode);
	}

	return output;
}
