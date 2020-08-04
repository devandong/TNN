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

#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/acc/metal_layer_acc.h"
#include "tnn/device/metal/metal_context.h"

namespace TNN_NS {

//DECLARE_METAL_ACC(Pooling, LAYER_POOLING);
class MetalPoolingLayerAcc : public MetalLayerAcc {
public:                                                                                                            
    virtual ~MetalPoolingLayerAcc(){};                                                                     
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                 
    virtual Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);     
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                 
    virtual std::string KernelName(); 
    virtual Status ComputeThreadSize(const std::vector<Blob *> &inputs, 
                            const std::vector<Blob *> &outputs, 
                            MTLSize &size); 
    virtual Status SetKernelEncoderParam(id<MTLComputeCommandEncoder> encoder, 
                                const std::vector<Blob *> &inputs, 
                                const std::vector<Blob *> &outputs);
private:
    bool use_global_pooling_ = false;
};

Status MetalPoolingLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto pool_param = dynamic_cast<PoolingLayerParam *>(param_);
    if (!pool_param || (pool_param->pool_type != 0 && pool_param->pool_type != 1)) {
        LOGE("Error: PoolingLayerParam pool_type unsupported\n");
        return Status(TNNERR_PARAM_ERR, "Error: PoolingLayerParam pool_type unsupported");
    }

    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalPoolingLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    id<MTLDevice> device          = [TNNMetalDeviceImpl sharedDevice];
    PoolingLayerParam *pool_param = dynamic_cast<PoolingLayerParam *>(param_);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // check if global_pooling
    use_global_pooling_ = (pool_param->kernels[0] == dims_input[3]) && \
                            (pool_param->kernels[1] == dims_input[2] && \
                            (pool_param->pads[0] == 0) && \
                            (pool_param->pads[2] == 0));
    //LOGE("\nis_global_pooling:%d, \n", use_global_pooling_);
    // buffer_param_
    {
        MetalPoolParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);

        metal_params.kernel_x = pool_param->kernels[0];
        metal_params.kernel_y = pool_param->kernels[1];
        metal_params.stride_x = pool_param->strides[0];
        metal_params.stride_y = pool_param->strides[1];
        metal_params.pad_x    = pool_param->pads[0];
        metal_params.pad_y    = pool_param->pads[2];

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalPoolParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

std::string MetalPoolingLayerAcc::KernelName() {
    auto param = dynamic_cast<PoolingLayerParam *>(param_);
    const int pool_type = param->pool_type;
    return pool_type == 0 ? "pooling_max" : use_global_pooling_ ? "pooling_global_sharedmemory" : "pooling_avg";
    //return pool_type == 0 ? "pooling_max" : "pooling_avg";
}

Status MetalPoolingLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<PoolingLayerParam *>(param_);
    if (!param || (param->pool_type != 0 && param->pool_type != 1)) {
        LOGE("Error: PoolingLayerParam pool_type unsupported\n");
        return Status(TNNERR_PARAM_ERR, "Error: PoolingLayerParam pool_type unsupported");
    }
    // global average pooling
    if (use_global_pooling_ && param->pool_type == 1) {
        //LOGE("use specialized global average pooling kernel!\n");
        Status status = TNN_OK;
        auto output = outputs[0];
        auto dims_output  = output->GetBlobDesc().dims;
        auto output_slice = UP_DIV(dims_output[1], 4);
        auto batch        = dims_output[0];
        
        auto context_impl = context_->getMetalContextImpl();
        auto encoder = [context_impl encoder];
        if (!encoder) {
            LOGE("encoder is nil\n");
            return Status(TNNERR_CONTEXT_ERR, "global average pooling encoder is nil");
        }
        encoder.label = [NSString stringWithFormat:@"layer: %s ", param_->name.c_str()];
        auto kernel_name = KernelName();
        if (kernel_name.length() <= 0) {
            status = Status(TNNERR_LAYER_ERR, "empty kernel name");
            return status;
        }
        MetalBandwidth bandwidth;
        do {
            status = [context_impl load:[NSString stringWithUTF8String:kernel_name.c_str()]
                            encoder:encoder
                          bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
            
            status = SetKernelEncoderParam(encoder, inputs, outputs);
        
            [encoder dispatchThreadgroups:{(NSUInteger)1, (NSUInteger)1, (NSUInteger)batch * output_slice}
                threadsPerThreadgroup:{(NSUInteger)32, (NSUInteger)1, (NSUInteger)1}];
            
            BREAK_IF(status != TNN_OK);
        } while(0);
        
        [encoder endEncoding];
        [context_impl commit];
        TNN_PRINT_ENCODER(context_, encoder, this);
        return status;
    }
    return MetalLayerAcc::Forward(inputs, outputs);
}

Status MetalPoolingLayerAcc::SetKernelEncoderParam(
                                               id<MTLComputeCommandEncoder> encoder,
                                               const std::vector<Blob *> &inputs,
                                               const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

Status MetalPoolingLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    size = GetDefaultThreadSize(dims_output, false);
    return TNN_OK;
}

REGISTER_METAL_ACC(Pooling, LAYER_POOLING);

} // namespace TNN_NS
