/************************************************************************************
***
***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2021-01-13 21:19:01
***
************************************************************************************/

#ifndef _GRID_SAMPLE_H
#define _GRID_SAMPLE_H

#include <onnxruntime_cxx_api.h>

struct GridSampleKernel {
	GridSampleKernel(OrtApi api, const OrtKernelInfo * info);

	void Compute(OrtKernelContext * context);

  protected:
	 OrtApi api_;
	 Ort::CustomOpApi ort_;
	const OrtKernelInfo *info_;
	 Ort::AllocatorWithDefaultOptions allocator_;

	int64_t align_corners_;
	int64_t interpolation_mode_;
	int64_t padding_mode_;
};

struct GridSampleOp:Ort::CustomOpBase < GridSampleOp, GridSampleKernel > {
	void *CreateKernel(OrtApi api, const OrtKernelInfo * info) const {
		return new GridSampleKernel(api, info);
	};

	const char *GetName() const {
		return "grid_sampler";
	};

	size_t GetInputTypeCount() const {
		return 2;
	};
	ONNXTensorElementDataType GetInputType(size_t /*index */ ) const {
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
	};

	size_t GetOutputTypeCount() const {
		return 1;
	};
	ONNXTensorElementDataType GetOutputType(size_t /*index */ ) const {
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
	};

	const char *GetExecutionProviderType() const {
		return "CPUExecutionProvider";
	};
};

#endif
