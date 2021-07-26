/************************************************************************************
***
***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2021-01-13 21:19:01
***
************************************************************************************/

#ifndef _DCNV2_FORWARD_H
#define _DCNV2_FORWARD_H

#include <onnxruntime_cxx_api.h>

// input, weight, bias, offset, mask,
// stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
// deformable_groups

struct DCNv2ForwardKernel {
  DCNv2ForwardKernel(OrtApi api, const OrtKernelInfo *info);

  void Compute(OrtKernelContext *context);

protected:
  OrtApi api_;
  Ort::CustomOpApi ort_;
  const OrtKernelInfo *info_;
  Ort::AllocatorWithDefaultOptions allocator_;

  // int64_t align_corners_;
  // int64_t interpolation_mode_;
  // int64_t padding_mode_;
};

struct DCNv2ForwardOp : Ort::CustomOpBase<DCNv2ForwardOp, DCNv2ForwardKernel> {
  void *CreateKernel(OrtApi api, const OrtKernelInfo *info) const {
    return new DCNv2ForwardKernel(api, info);
  };

  const char *GetName() const { return "dcnv2_forward"; };

  size_t GetInputTypeCount() const { return 12; };
  ONNXTensorElementDataType GetInputType(size_t /*index */) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index */) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  const char *GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  };
};

#endif
