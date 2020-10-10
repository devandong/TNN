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

#include <metal_stdlib>
#include "tnn/device/metal/acc/metal_common.metal"

using namespace metal;

kernel void inner_product(const device ftype4 *in                                      [[buffer(0)]],
                                         device ftype4 *out                                              [[buffer(1)]],
                                         constant MetalInnerProductParams & params  [[buffer(2)]],
                                         const device ftype4x4 *wt                                  [[buffer(3)]],
                                         const device ftype4 *biasTerms                         [[buffer(4)]],
                                         uint3 gid                                                             [[thread_position_in_grid]]) {
    if ((int)gid.x >= params.output_size || (int)gid.y >= params.output_slice || (int)gid.z >= params.batch) return;
    
    auto xy_wt  = wt                                                    + (int)gid.y * params.input_slice * params.input_size;
    auto xy_in  = in  + (int)gid.z * params.input_slice  * params.input_size  + (int)gid.x;
    auto xy_out = out + (int)gid.z * params.output_slice * params.output_size + (int)gid.y * params.output_size + (int)gid.x;
    
    auto result = params.has_bias ? float4(biasTerms[gid.y]) : float4(Zero4);
    for (auto z = 0; z < params.input_slice*params.input_size; z++) {
            result += float4(xy_in[z]) * float4x4(xy_wt[z]);
    }
    *xy_out = activate(ftype4(result), params.activation);
}

#define VEC_K
// finer-grained inner-product kernel
kernel void inner_product_fg(const device ftype *in                               [[buffer(0)]],
                                      device ftype *out                           [[buffer(1)]],
                                constant MetalInnerProductParams & params         [[buffer(2)]],
                                const device ftype *wt                            [[buffer(3)]],
                                const device ftype *biasTerms                     [[buffer(4)]],
                                uint3 gid                              [[thread_position_in_grid]]) {
    // each thread responsible for one output element
    if ((int)gid.x >= params.output_size || (int)gid.y >= params.output_channel || (int)gid.z >= params.batch) return;

    const int input_channel  = params.input_slice * 4;
    const int output_channel = params.output_slice * 4;

    const int output_slice_idx   = (int)gid.y / 4;
    const int output_channel_idx = (int)gid.y - output_slice_idx * 4;
#ifdef VEC_K
    const device ftype4 * xy_wt  = reinterpret_cast<const device ftype4 *>(wt  + (int)gid.y * input_channel * params.input_size);
    const device ftype4 * xy_in  = reinterpret_cast<const device ftype4 *>(in  + (int)gid.z * input_channel * params.input_size + (int)gid.x);
    device ftype *xy_out = out + (int)gid.z * output_channel * params.output_size + (output_slice_idx * params.output_size + (int)gid.x) * 4 + output_channel_idx;

    float4 result4 = float4(Zero4);
    for (auto z = 0; z < params.input_slice * params.input_size; z++) {
        result4 += float4(xy_in[z]) * float4(xy_wt[z]);
    }

    float result = params.has_bias ? float(biasTerms[gid.y]) : float(0);
    // horizontal reduction
    result += result4.x + result4.y + result4.z + result4.w;
#else
    const device ftype * xy_wt  = wt  + (int)gid.y * input_channel * params.input_size;
    const device ftype * xy_in  = in  + (int)gid.z * input_channel * params.input_size + (int)gid.x;
    device ftype *xy_out = out + (int)gid.z * output_channel * params.output_size + (output_slice_idx * params.output_size + (int)gid.x) * 4 + output_channel_idx;

    float result = params.has_bias ? float(biasTerms[gid.y]) : float(0);
    for (auto z = 0; z < input_channel * params.input_size; z++) {
        result += float(xy_in[z]) * float(xy_wt[z]);
    }
#endif
    *xy_out = activate(ftype(result), params.activation);
}

#define THREADGROUP_SIZE 128
#define kNumSplit 4

kernel void inner_product_fg_splitk(const device ftype *in                               [[buffer(0)]],
                                      device ftype *out                           [[buffer(1)]],
                                constant MetalInnerProductParams & params         [[buffer(2)]],
                                const device ftype *wt                            [[buffer(3)]],
                                const device ftype *biasTerms                     [[buffer(4)]],
                                uint id                               [[thread_position_in_grid]],
                                uint tid                              [[thread_index_in_threadgroup]],
                                uint bid                              [[threadgroup_position_in_grid]],
                                uint blockDim                         [[threads_per_threadgroup]]) {
    // shared memory for reduction
    threadgroup float shared_memory[THREADGROUP_SIZE];
    shared_memory[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int output_channel_per_threadgroup = THREADGROUP_SIZE / kNumSplit;
    const int output_channel_within_threadgroup = (int)tid % output_channel_per_threadgroup;
    const int output_channel_idx = bid * output_channel_per_threadgroup + output_channel_within_threadgroup;
    const int output_slice_idx = output_channel_idx / 4;
    const int output_channel_offset = output_channel_idx - output_slice_idx * 4;

    const int input_channel_split_idx = tid / output_channel_per_threadgroup;
    const int input_slice_per_split = (params.input_slice + kNumSplit - 1) / kNumSplit;
    const int input_slice_idx = input_channel_split_idx * input_slice_per_split;
    const int input_channel_idx = input_slice_idx * 4;

    const int gid = bid * output_channel_per_threadgroup + output_channel_within_threadgroup;
    const int output_spatial_idx = (int)gid % params.output_size;
    const int output_channel_updiv = ROUND_UP(params.output_channel, output_channel_per_threadgroup);
    const int batch_idx = (int)gid / params.output_size / output_channel_updiv;

    const int input_channel4  = params.input_slice * 4;
    const int output_channel4 = params.output_slice * 4;

    // compute the ptr for input, weight and output for each thread
    const device ftype4 * xy_wt  = reinterpret_cast<const device ftype4 *>(wt  + (output_channel_idx * input_channel4 + input_channel_idx) * params.input_size);
    const device ftype4 * xy_in  = reinterpret_cast<const device ftype4 *>(in  + (batch_idx * input_channel4 + input_channel_idx) * params.input_size + output_spatial_idx);
    device ftype *xy_out = out + batch_idx * output_channel4 * params.output_size + (output_slice_idx * params.output_size + output_spatial_idx) * 4 + output_channel_offset;

    float result = 0;
    if (output_channel_idx < params.output_channel) {
        float4 result4 = float4(Zero4);
        const int kNumLeftInputSlice = min(input_slice_per_split, params.input_slice - input_slice_idx);
        for (auto z = 0; z < kNumLeftInputSlice * params.input_size; z++) {
            result4 += float4(xy_in[z]) * float4(xy_wt[z]);
        }
        // horizontal reduction
        result = result4.x + result4.y + result4.z + result4.w;
        shared_memory[tid] = result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(tid < output_channel_per_threadgroup && output_channel_idx < params.output_channel) {
        float merged = params.has_bias? float(biasTerms[output_channel_idx]) : float(0);
        for(int i=tid; i < THREADGROUP_SIZE; i+=output_channel_per_threadgroup) {
            merged += shared_memory[i];
        }
        *xy_out = activate(merged, params.activation);;
    }
}
