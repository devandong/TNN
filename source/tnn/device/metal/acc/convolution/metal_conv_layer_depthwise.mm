// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tnn/device/metal/acc/convolution/metal_conv_layer_depthwise.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils.h"

namespace TNN_NS {
bool MetalConvLayerDepthwise::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    auto input_dims          = inputs[0]->GetBlobDesc().dims;
    auto output_dims         = outputs[0]->GetBlobDesc().dims;

    return param->group == input_dims[1] && param->group == output_dims[1];
}

MetalConvLayerDepthwise::~MetalConvLayerDepthwise() {}

Status MetalConvLayerDepthwise::AllocateBufferWeight(const std::vector<Blob *> &inputs,
                                                     const std::vector<Blob *> &outputs) {
    auto param  = dynamic_cast<ConvLayerParam *>(param_);
    auto resource = dynamic_cast<ConvLayerResource *>(resource_);

    Status status = TNN_OK;
    if (!buffer_weight_) {
        int kw = param->kernels[0];
        int kh = param->kernels[1];

        const int group = param->group;
        buffer_weight_ =  AllocatePackedNC4HW4MetalBufferFormRawBuffer(
                                                                       resource->filter_handle,
                                                                       {1, group, kh, kw},
                                                                       group,
                                                                       status);
    }
    return status;
}

Status MetalConvLayerDepthwise::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                    const std::vector<Blob *> &outputs) {
    id<MTLDevice> device        = [TNNMetalDeviceImpl sharedDevice];
    ConvLayerParam *layer_param = dynamic_cast<ConvLayerParam *>(param_);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalConvParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);
        SetDefaultMetalConvParams(metal_params, layer_param);

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalConvParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    // check if specialized kernels should be used
    bool s11 = (layer_param->strides[0] == 1 && layer_param->strides[1] == 1);
    bool s22 = (layer_param->strides[0] == 2 && layer_param->strides[1] == 2);
    bool d11 = (layer_param->dialations[0] == 1 && layer_param->dialations[1] == 1);
    bool k33 = (layer_param->kernels[0] == 3 && layer_param->kernels[1] == 3);
    bool k55 = (layer_param->kernels[0] == 5 && layer_param->kernels[1] == 5);
    bool k15 = (layer_param->kernels[0] == 1 && layer_param->kernels[1] == 5);
    bool k51 = (layer_param->kernels[0] == 5 && layer_param->kernels[1] == 1);
    if (s11 && d11 && k33) {
        this->k3s1d1_spec = true;
    } else if (s22 && d11 && k33) {
        this->k3s2d1_spec = true;
    } else if (s11 && d11 && k55) {
        this->k5s1d1_spec = true;
    } else if (s11 && d11 && k51) {
        this->k51s1d1_spec = true;
    } else if (s11 && d11 && k15) {
        this->k15s1d1_spec = true;
    }

    return TNN_OK;
}

std::string MetalConvLayerDepthwise::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (k3s1d1_spec) {
        //return "convolution_depthwise3x3";
        return "convolution_depthwise3x3_h8w4";
        //return "convolution_depthwise3x3_h8w8";
    }
    if (k5s1d1_spec) {
        //printf("%s\n", "==== specialized depthwise 5x5 kernel!=====");
        return "convolution_depthwise5x5_h8w4";
    }
    if (k51s1d1_spec) {
        //printf("====== 5x1 conv kernel!\n");
        return "convolution_depthwise5x1_h8w4";
    }
    if (k15s1d1_spec) {
        //printf("====== 1x5 conv kernel!\n");
        return "convolution_depthwise1x5_h4w8";
    }
    return "convolution_depthwise";
}

Status MetalConvLayerDepthwise::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
    [encoder setBuffer:buffer_weight_
                offset:0
               atIndex:3];
    [encoder setBuffer:buffer_bias_
                offset:0
               atIndex:4];
    return TNN_OK;
}

Status MetalConvLayerDepthwise::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto output = outputs[0];
    auto dims_output  = output->GetBlobDesc().dims;
    auto output_height  = dims_output[2];
    auto output_width  = dims_output[3];
    auto output_slice = UP_DIV(dims_output[1], 4);
    auto output_batch = dims_output[0];
    if (k3s1d1_spec) {
        size = MTLSizeMake(4, 8, 1);
        //size = MTLSizeMake(UP_DIV(output_width, 2),
        //                   UP_DIV(output_height, 2),
        //                   output_slice*output_batch);
    } else if (k5s1d1_spec) {
        size = MTLSizeMake(4, 8, 1);
    } else if (k51s1d1_spec) {
        size = MTLSizeMake(4, 8, 1);
    } else if (k15s1d1_spec) {
        size = MTLSizeMake(8, 4, 1);
    } else {
        size = GetDefaultThreadSize(dims_output, false);
    }
    return TNN_OK;
}

Status MetalConvLayerDepthwise::ComputeThreadgroupSize(const std::vector<Blob *> &inputs,
                                     const std::vector<Blob *> &outputs,
                                     MTLSize &size) {
    auto dims_output  = outputs[0]->GetBlobDesc().dims;
    auto output_height = dims_output[2];
    auto output_width  = dims_output[3];
    auto output_slice = UP_DIV(dims_output[1], 4);
    auto output_batch = dims_output[0];
    if (k3s1d1_spec) {
        size = MTLSizeMake(UP_DIV(output_width, 4),
                          UP_DIV(output_height, 8),
                          output_batch*output_slice);
        //size = MTLSizeMake(0, 0, 0);
    } else if (k5s1d1_spec) {
        size = MTLSizeMake(UP_DIV(output_width, 4),
                          UP_DIV(output_height, 8),
                          output_batch*output_slice);
    } else if (k51s1d1_spec) {
        size = MTLSizeMake(UP_DIV(output_width, 4),
                           UP_DIV(output_height, 8),
                           output_batch*output_slice);
    } else if (k15s1d1_spec) {
        size = MTLSizeMake(UP_DIV(output_width, 8),
                           UP_DIV(output_height, 4),
                           output_batch*output_slice);
    } else {
        size = MTLSizeMake(0, 0, 0);
    }
    return TNN_OK;
}

Status MetalConvLayerDepthwise::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

} // namespace TNN_NS
