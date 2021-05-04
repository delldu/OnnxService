/************************************************************************************
***
*** Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
*** File Author: Dell, 2021-01-12 23:52:44
***
************************************************************************************/

// This file comes from: 
// https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/onnxruntime/cpu/gridSample.cpp
// Thanks a lot.

#include <math.h>
#include <vector>

#include "dcnv2_forward.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) < (b)) ? (b) : (a))

// modified from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/GridSampler.cpp

struct OrtTensorDimensions:std::vector < int64_t > {
	OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue * value) {
		OrtTensorTypeAndShapeInfo *info = ort.GetTensorTypeAndShape(value);
		std::vector < int64_t >::operator=(ort.GetTensorShape(info));
		ort.ReleaseTensorTypeAndShapeInfo(info);
}};

DCNv2ForwardKernel::DCNv2ForwardKernel(OrtApi api, const OrtKernelInfo * info)
:api_(api), ort_(api_), info_(info)
{
	// align_corners_ = ort_.KernelInfoGetAttribute < int64_t > (info, "align_corners");
	// interpolation_mode_ = ort_.KernelInfoGetAttribute < int64_t > (info, "interpolation_mode");
	// padding_mode_ = ort_.KernelInfoGetAttribute < int64_t > (info, "padding_mode");

	allocator_ = Ort::AllocatorWithDefaultOptions();
}

void DCNv2ForwardKernel::Compute(OrtKernelContext * context)
{
	// const bool align_corners = align_corners_;
	// const int64_t padding_mode = padding_mode_;
	// const int64_t interpolation_mode = interpolation_mode_;

	const OrtValue *input = ort_.KernelContext_GetInput(context, 0);
	const float *input_data = reinterpret_cast < const float *>(ort_.GetTensorData < float >(input));

	const OrtValue *grid = ort_.KernelContext_GetInput(context, 1);
	const float *grid_data = reinterpret_cast < const float *>(ort_.GetTensorData < float >(grid));

	OrtTensorDimensions input_dims(ort_, input);
	OrtTensorDimensions grid_dims(ort_, grid);
	int64_t N = input_dims[0];
	int64_t C = input_dims[1];
	int64_t inp_H = input_dims[2];
	int64_t inp_W = input_dims[3];
	int64_t out_H = grid_dims[1];
	int64_t out_W = grid_dims[2];

	std::vector < int64_t > output_dims = {
	N, C, out_H, out_W};
	OrtValue *output = ort_.KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size());
	float *out_ptr = ort_.GetTensorMutableData < float >(output);


	int64_t out_sN = output_dims[1] * output_dims[2] * output_dims[3];
	int64_t out_sC = output_dims[2] * output_dims[3];
	int64_t out_sH = output_dims[3];
	int64_t out_sW = 1;
}

