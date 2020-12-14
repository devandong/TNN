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

#define TRANS_WEIGHT 0

using namespace metal;

kernel void convolution_depthwise(const device ftype4 *in           [[buffer(0)]],
                                  device ftype4 *out                [[buffer(1)]],
                                  constant MetalConvParams& params  [[buffer(2)]],
                                  const device ftype4 *wt           [[buffer(3)]],
                                  const device ftype4 *biasTerms    [[buffer(4)]],
                                  uint3 gid                       [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width,
                           params.output_height,
                           params.output_slice)))
        return;
    
//    short oz = gid.z % params.output_slice;
    int oz = gid.z;
    int offset_x = (int)gid.x * params.stride_x - params.pad_x;
    int offset_y = (int)gid.y * params.stride_y - params.pad_y;
    int sx = max(0, (UP_DIV(-offset_x, params.dilation_x)));
    int ex = min(params.kernel_x, UP_DIV(params.input_width - offset_x, params.dilation_x));
    int sy = max(0, (UP_DIV(-offset_y, params.dilation_y)));
    int ey = min(params.kernel_y, UP_DIV(params.input_height - offset_y, params.dilation_y));
    offset_x += sx * params.dilation_x;
    offset_y += sy * params.dilation_y;
    
    auto z_wt  = wt  + (int)oz * params.kernel_size;
    auto z_in  = in  + (int)gid.z * params.input_size;
    auto z_out = out + (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
    
    auto result = params.has_bias ? float4(biasTerms[oz]) : float4(Zero4);
    for (auto ky = sy, y = offset_y; ky < ey; ky++, y += params.dilation_y) {
        for (auto kx = sx, x = offset_x; kx < ex; kx++, x += params.dilation_x) {
            auto wt4 = float4(z_wt[ky * params.kernel_x   + kx]);
            auto in4 = float4(z_in[ y * params.input_width + x]);
            result += in4 * wt4;
        }
    }
    
    *z_out = activate(ftype4(result), params.activation);
}

kernel void convolution_depthwise3x3(const device ftype4 *in           [[buffer(0)]],
                                          device ftype4 *out                [[buffer(1)]],
                                          constant MetalConvParams& params  [[buffer(2)]],
                                          const device ftype4 *wt           [[buffer(3)]],
                                          const device ftype4 *biasTerms    [[buffer(4)]],
                                          uint3 gid                       [[thread_position_in_grid]]) {
    int gid_x = gid.x * 2;
    int gid_y = gid.y * 2;
    int gid_z = gid.z;
    
    if (gid_x >= params.output_width || gid_y >= params.output_height) {
        return;
    }
    
    float4 r0 = params.has_bias? float4(biasTerms[gid_z]) : float4(Zero4);
    float4 l0 = params.has_bias? float4(biasTerms[gid_z]) : float4(Zero4);
    float4 t0 = params.has_bias? float4(biasTerms[gid_z]) : float4(Zero4);
    float4 b0 = params.has_bias? float4(biasTerms[gid_z]) : float4(Zero4);
    
    int x0 = gid_x - params.pad_x;
    int x1 = gid_x - params.pad_x + 1;
    int x2 = gid_x - params.pad_x + 2;
    int x3 = gid_x - params.pad_x + 3;
    int y0 = gid_y - params.pad_y;
    int y1 = gid_y - params.pad_y + 1;
    int y2 = gid_y - params.pad_y + 2;
    int y3 = gid_y - params.pad_y + 3;
    
    bool x0_out = x0 < 0 || x0 >= params.input_width;
    bool x1_out = x1 < 0 || x1 >= params.input_width;
    bool x2_out = x2 < 0 || x2 >= params.input_width;
    bool x3_out = x3 < 0 || x3 >= params.input_width;
    bool y0_out = y0 < 0 || y0 >= params.input_height;
    bool y1_out = y1 < 0 || y1 >= params.input_height;
    bool y2_out = y2 < 0 || y2 >= params.input_height;
    bool y3_out = y3 < 0 || y3 >= params.input_height;
    
    x0 = clamp(x0, 0, params.input_width - 1);
    x1 = clamp(x1, 0, params.input_width - 1);
    x2 = clamp(x2, 0, params.input_width - 1);
    x3 = clamp(x3, 0, params.input_width - 1);
    y0 = clamp(y0, 0, params.input_height - 1);
    y1 = clamp(y1, 0, params.input_height - 1);
    y2 = clamp(y2, 0, params.input_height - 1);
    y3 = clamp(y3, 0, params.input_height - 1);
    
    device const ftype4* src_loc = in + gid_z * params.input_size;
    device const ftype4* filters_loc = wt + gid_z * 9;
    
    ftype4 s0 = src_loc[y0 * params.input_width + x0] * ftype(!(x0_out || y0_out));
    ftype4 s1 = src_loc[y1 * params.input_width + x0] * ftype(!(x0_out || y1_out));
    ftype4 s2 = src_loc[y2 * params.input_width + x0] * ftype(!(x0_out || y2_out));
    ftype4 s3 = src_loc[y3 * params.input_width + x0] * ftype(!(x0_out || y3_out));
    
    r0 += float4(s0 * filters_loc[0]);
    r0 += float4(s1 * filters_loc[3]);
    r0 += float4(s2 * filters_loc[6]);
    l0 += float4(s1 * filters_loc[0]);
    l0 += float4(s2 * filters_loc[3]);
    l0 += float4(s3 * filters_loc[6]);

    s0 = src_loc[y0 * params.input_width + x1] * ftype(!(x1_out || y0_out));
    s1 = src_loc[y1 * params.input_width + x1] * ftype(!(x1_out || y1_out));
    s2 = src_loc[y2 * params.input_width + x1] * ftype(!(x1_out || y2_out));
    s3 = src_loc[y3 * params.input_width + x1] * ftype(!(x1_out || y3_out));

    r0 += float4(s0 * filters_loc[1]);
    r0 += float4(s1 * filters_loc[4]);
    r0 += float4(s2 * filters_loc[7]);
    l0 += float4(s1 * filters_loc[1]);
    l0 += float4(s2 * filters_loc[4]);
    l0 += float4(s3 * filters_loc[7]);
    t0 += float4(s0 * filters_loc[0]);
    t0 += float4(s1 * filters_loc[3]);
    t0 += float4(s2 * filters_loc[6]);
    b0 += float4(s1 * filters_loc[0]);
    b0 += float4(s2 * filters_loc[3]);
    b0 += float4(s3 * filters_loc[6]);
    
    s0 = src_loc[y0 * params.input_width + x2] * ftype(!(x2_out || y0_out));
    s1 = src_loc[y1 * params.input_width + x2] * ftype(!(x2_out || y1_out));
    s2 = src_loc[y2 * params.input_width + x2] * ftype(!(x2_out || y2_out));
    s3 = src_loc[y3 * params.input_width + x2] * ftype(!(x2_out || y3_out));
    
    r0 += float4(s0 * filters_loc[2]);
    r0 += float4(s1 * filters_loc[5]);
    r0 += float4(s2 * filters_loc[8]);
    l0 += float4(s1 * filters_loc[2]);
    l0 += float4(s2 * filters_loc[5]);
    l0 += float4(s3 * filters_loc[8]);
    t0 += float4(s0 * filters_loc[1]);
    t0 += float4(s1 * filters_loc[4]);
    t0 += float4(s2 * filters_loc[7]);
    b0 += float4(s1 * filters_loc[1]);
    b0 += float4(s2 * filters_loc[4]);
    b0 += float4(s3 * filters_loc[7]);
    
    s0 = src_loc[y0 * params.input_width + x3] * ftype(!(x3_out || y0_out));
    s1 = src_loc[y1 * params.input_width + x3] * ftype(!(x3_out || y1_out));
    s2 = src_loc[y2 * params.input_width + x3] * ftype(!(x3_out || y2_out));
    s3 = src_loc[y3 * params.input_width + x3] * ftype(!(x3_out || y3_out));
    
    t0 += float4(s0 * filters_loc[2]);
    t0 += float4(s1 * filters_loc[5]);
    t0 += float4(s2 * filters_loc[8]);
    b0 += float4(s1 * filters_loc[2]);
    b0 += float4(s2 * filters_loc[5]);
    b0 += float4(s3 * filters_loc[8]);

    const int offset_0 = gid_z * params.output_size + gid_y * params.output_width + gid_x;
    const int offset_1 = offset_0 + params.output_width;
    const int offset_2 = offset_0 + 1;
    const int offset_3 = offset_0 + params.output_width + 1;
    bool x0_in = gid_x < params.output_width;
    bool x1_in = gid_x + 1 < params.output_width;
    bool y0_in = gid_y < params.output_height;
    bool y1_in = gid_y + 1 < params.output_height;
    
    
    if (y0_in && x0_in) {
        out[offset_0] = activate(ftype4(r0), params.activation);
    }
    if (y1_in && x0_in) {
        out[offset_1] = activate(ftype4(l0), params.activation);
    }
    if (y0_in && x1_in) {
        out[offset_2] = activate(ftype4(t0), params.activation);
    }
    if (y1_in && x1_in) {
        out[offset_3] = activate(ftype4(b0), params.activation);
    }
}

kernel void convolution_depthwise3x3_h8w4(const device ftype4 *in           [[buffer(0)]],
                                          device ftype4 *out                [[buffer(1)]],
                                          constant MetalConvParams& params  [[buffer(2)]],
                                          const device ftype4 *wt           [[buffer(3)]],
                                          const device ftype4 *biasTerms    [[buffer(4)]],
                                          uint3 gid                       [[thread_position_in_grid]],
                                          uint3 group_id                  [[threadgroup_position_in_grid]],
                                          uint thread_index               [[thread_index_in_threadgroup]]) {
    threadgroup ftype4 input_data_cache[6 * 10];
    
    // compute ld offset of inputs
    const int ld_start_w = group_id.x * 4 - params.pad_x;
    const int ld_start_h = group_id.y * 8 - params.pad_y;
    const int ld_start_c = group_id.z;
    
    const int ld_offset = ld_start_c * params.input_size;
    
    const int a_smem_st_offset = thread_index;
    
    // load data
    int ld_w = ld_start_w + thread_index % 6;
    int ld_h = ld_start_h + thread_index / 6;
    int ld_pos = ld_offset + ld_h * params.input_width + ld_w;
    bool w_in_image = ld_w >=0 && ld_w < params.input_width;
    bool in_image = (ld_h >=0 && ld_h < params.input_height) && w_in_image;
    ftype4 v = in_image ? in[ld_pos] : Zero4;
    input_data_cache[a_smem_st_offset] = v;
    
    ld_w = ld_start_w + (thread_index + 32) % 6;
    ld_h = ld_start_h + (thread_index + 32) / 6;
    w_in_image = ld_w >=0 && ld_w < params.input_width;
    in_image =  (ld_h >= 0 && ld_h < params.input_height) && w_in_image ;
    ld_pos = ld_offset + ld_h * params.input_width + ld_w;
    v = in_image ? in[ld_pos] : Zero4;
    if (a_smem_st_offset + 32 < 60)
        input_data_cache[a_smem_st_offset +  32] = v;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (!any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch))) {
        auto z_out = out + (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
        auto result = params.has_bias ? biasTerms[gid.z] : Zero4;
        auto z_wt  = wt  + (int)gid.z * params.kernel_size;
        
        int offset_x = thread_index % 4;
        int offset_y = thread_index / 4;
#pragma unroll
        for (auto ky = 0, y = offset_y; ky < 3; ky++, y ++) {
            for (auto kx = 0, x = offset_x; kx < 3; kx++, x ++) {
                auto wt4 = z_wt[ky * 3   + kx];
                auto in4 = input_data_cache[ y * 6 + x];
                result += in4 * wt4;
            }
        }
        
        *z_out = activate(result, params.activation);
    }
}

kernel void convolution_depthwise3x3_h8w8(const device ftype4 *in           [[buffer(0)]],
                                          device ftype4 *out                [[buffer(1)]],
                                          constant MetalConvParams& params  [[buffer(2)]],
                                          const device ftype4 *wt           [[buffer(3)]],
                                          const device ftype4 *biasTerms    [[buffer(4)]],
                                          uint3 gid                       [[thread_position_in_grid]],
                                          uint3 group_id                  [[threadgroup_position_in_grid]],
                                          uint thread_index               [[thread_index_in_threadgroup]]) {
    threadgroup ftype4 input_data_cache[10 * 10];
    
    // compute ld offset of inputs
    const int ld_start_w = group_id.x * 8 - params.pad_x;
    const int ld_start_h = group_id.y * 8 - params.pad_y;
    const int ld_start_c = group_id.z;
    
    const int ld_offset = ld_start_c * params.input_size;
    
    const int a_smem_st_offset = thread_index;
    
    // load data
    int ld_w = ld_start_w + thread_index % 10;
    int ld_h = ld_start_h + thread_index / 10;
    int ld_pos = ld_offset + ld_h * params.input_width + ld_w;
    bool w_in_image = ld_w >=0 && ld_w < params.input_width;
    bool in_image = (ld_h >=0 && ld_h < params.input_height) && w_in_image;
    ftype4 v = in_image ? in[ld_pos] : Zero4;
    input_data_cache[a_smem_st_offset] = v;
    
    ld_w = ld_start_w + (thread_index + 32) % 10;
    ld_h = ld_start_h + (thread_index + 32) / 10;
    w_in_image = ld_w >=0 && ld_w < params.input_width;
    in_image =  (ld_h >= 0 && ld_h < params.input_height) && w_in_image ;
    ld_pos = ld_offset + ld_h * params.input_width + ld_w;
    v = in_image ? in[ld_pos] : Zero4;
    input_data_cache[a_smem_st_offset +  32] = v;
    
    ld_w = ld_start_w + (thread_index + 64) % 10;
    ld_h = ld_start_h + (thread_index + 64) / 10;
    w_in_image = ld_w >=0 && ld_w < params.input_width;
    in_image =  (ld_h >= 0 && ld_h < params.input_height) && w_in_image ;
    ld_pos = ld_offset + ld_h * params.input_width + ld_w;
    v = in_image ? in[ld_pos] : Zero4;
    input_data_cache[a_smem_st_offset +  64] = v;
    
    ld_w = ld_start_w + (thread_index + 96) % 10;
    ld_h = ld_start_h + (thread_index + 96) / 10;
    w_in_image = ld_w >=0 && ld_w < params.input_width;
    in_image =  (ld_h >= 0 && ld_h < params.input_height) && w_in_image ;
    ld_pos = ld_offset + ld_h * params.input_width + ld_w;
    v = in_image ? in[ld_pos] : Zero4;
    if (a_smem_st_offset + 96 < 100)
        input_data_cache[a_smem_st_offset +  96] = v;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    auto z_wt  = wt  + (int)gid.z * params.kernel_size;
    
    int x0 = (thread_index % 4) * 2;
    int x1 = x0 + 1;
    int x2 = x0 + 2;
    int x3 = x0 + 3;
    int y0 = thread_index / 4;
    int y1 = y0 + 1;
    int y2 = y0 + 2;
    
    float4 r1 = params.has_bias ? float4(biasTerms[gid.z]) : float4(Zero4);
    float4 r2 = params.has_bias ? float4(biasTerms[gid.z]) : float4(Zero4);
    
    ftype4 s00 = input_data_cache[y0 * 10 + x0];
    ftype4 s01 = input_data_cache[y0 * 10 + x1];
    ftype4 s02 = input_data_cache[y0 * 10 + x2];
    ftype4 s03 = input_data_cache[y0 * 10 + x3];
    
    r1 += float4(s00 * z_wt[0 * 3 + 0]);
    r2 += float4(s01 * z_wt[0 * 3 + 0]);
    r1 += float4(s01 * z_wt[0 * 3 + 1]);
    r2 += float4(s02 * z_wt[0 * 3 + 1]);
    r1 += float4(s02 * z_wt[0 * 3 + 2]);
    r2 += float4(s03 * z_wt[0 * 3 + 2]);
    
    ftype4 s10 = input_data_cache[y1 * 10 + x0];
    ftype4 s11 = input_data_cache[y1 * 10 + x1];
    ftype4 s12 = input_data_cache[y1 * 10 + x2];
    ftype4 s13 = input_data_cache[y1 * 10 + x3];
    
    r1 += float4(s10 * z_wt[1 * 3 + 0]);
    r2 += float4(s11 * z_wt[1 * 3 + 0]);
    r1 += float4(s11 * z_wt[1 * 3 + 1]);
    r2 += float4(s12 * z_wt[1 * 3 + 1]);
    r1 += float4(s12 * z_wt[1 * 3 + 2]);
    r2 += float4(s13 * z_wt[1 * 3 + 2]);
    
    ftype4 s20 = input_data_cache[y2 * 10 + x0];
    ftype4 s21 = input_data_cache[y2 * 10 + x1];
    ftype4 s22 = input_data_cache[y2 * 10 + x2];
    ftype4 s23 = input_data_cache[y2 * 10 + x3];
    
    r1 += float4(s20 * z_wt[2 * 3 + 0]);
    r2 += float4(s21 * z_wt[2 * 3 + 0]);
    r1 += float4(s21 * z_wt[2 * 3 + 1]);
    r2 += float4(s22 * z_wt[2 * 3 + 1]);
    r1 += float4(s22 * z_wt[2 * 3 + 2]);
    r2 += float4(s23 * z_wt[2 * 3 + 2]);
    
    const int out_s = gid.z;
    const int out_h = gid.y;
    const int out_w1 = gid.x * 2;
    const int out_w2 = out_w1 + 1;
    auto z_out1 = out + out_s * params.output_size + out_h * params.output_width + out_w1;
    auto z_out2 = out + out_s * params.output_size + out_h * params.output_width + out_w2;
    
    if (all(int3(out_w1, out_h, out_s) < int3(params.output_width, params.output_height, params.output_slice*params.batch))) {
        *z_out1 = activate(ftype4(r1), params.activation);
    }
    if (all(int3(out_w2, out_h, out_s) < int3(params.output_width, params.output_height, params.output_slice*params.batch))) {
        *z_out2 = activate(ftype4(r2), params.activation);
    }
}

kernel void convolution_depthwise5x5_h8w4(const device ftype4 *in           [[buffer(0)]],
                                          device ftype4 *out                [[buffer(1)]],
                                          constant MetalConvParams& params  [[buffer(2)]],
                                          const device ftype4 *wt           [[buffer(3)]],
                                          const device ftype4 *biasTerms    [[buffer(4)]],
                                          uint3 gid                       [[thread_position_in_grid]],
                                          uint3 group_id                  [[threadgroup_position_in_grid]],
                                          uint thread_index               [[thread_index_in_threadgroup]]) {
    threadgroup ftype4 input_data_cache[8 * 12];
    
    // compute ld offset of inputs
    const int ld_start_w = group_id.x * 4 - params.pad_x;
    const int ld_start_h = group_id.y * 8 - params.pad_y;
    const int ld_start_c = group_id.z;
    
    const int ld_offset = ld_start_c * params.input_size;
    
    const int a_smem_st_offset = thread_index;
    
    // load data
    int ld_w = ld_start_w + thread_index % 8;
    int ld_h = ld_start_h + thread_index / 8;
    const int ld_pos = ld_offset + ld_h * params.input_width + ld_w;
    
    bool w_in_image = ld_w >=0 && ld_w < params.input_width;
    
    bool in_image = (ld_h >=0 && ld_h < params.input_height) && w_in_image;
    ftype4 v = in_image ? in[ld_pos] : Zero4;
    input_data_cache[a_smem_st_offset] = v;
    
    bool in_image1 =  ld_h + 4 >= 0 && ld_h + 4 < params.input_height && w_in_image ;
    v = in_image1 ? in[ld_pos + 4 * params.input_width] : Zero4;
    input_data_cache[a_smem_st_offset +  32] = v;
    
    bool in_image2 =  ld_h + 8 >= 0 && ld_h + 8 < params.input_height && w_in_image;
    v = in_image2 ? in[ld_pos + 8 * params.input_width] : Zero4;
    input_data_cache[a_smem_st_offset + 64] = v;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (!any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch))) {
        auto z_out = out + (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
        auto result = params.has_bias ? biasTerms[gid.z] : Zero4;
        auto z_wt  = wt  + (int)gid.z * params.kernel_size;
        
        int offset_x = thread_index % 4;
        int offset_y = thread_index / 4;
#pragma unroll
        for (auto ky = 0, y = offset_y; ky < 5; ky++, y ++) {
            for (auto kx = 0, x = offset_x; kx < 5; kx++, x ++) {
                auto wt4 = z_wt[ky * 5   + kx];
                auto in4 = input_data_cache[ y * 8 + x];
                result += in4 * wt4;
            }
        }
        
        *z_out = activate(result, params.activation);
    }
}

kernel void convolution_depthwise5x1_h8w4(const device ftype4 *in           [[buffer(0)]],
                                          device ftype4 *out                [[buffer(1)]],
                                          constant MetalConvParams& params  [[buffer(2)]],
                                          const device ftype4 *wt           [[buffer(3)]],
                                          const device ftype4 *biasTerms    [[buffer(4)]],
                                          uint3 gid                       [[thread_position_in_grid]],
                                          uint3 group_id                  [[threadgroup_position_in_grid]],
                                          uint thread_index               [[thread_index_in_threadgroup]]) {
    threadgroup ftype4 input_data_cache[8 * 8];
    
    // compute ld offset of inputs
    const int ld_start_w = group_id.x * 4 - params.pad_x;
    const int ld_start_h = group_id.y * 8 - params.pad_y;
    const int ld_start_c = group_id.z;
    
    const int ld_offset = ld_start_c * params.input_size;
    
    const int a_smem_st_offset = thread_index;
    
    // load data
    int ld_w = ld_start_w + thread_index % 8;
    int ld_h = ld_start_h + thread_index / 8;
    const int ld_pos = ld_offset + ld_h * params.input_width + ld_w;
    
    bool w_in_image = ld_w >=0 && ld_w < params.input_width;
    
    bool in_image = (ld_h >=0 && ld_h < params.input_height) && w_in_image;
    ftype4 v = in_image ? in[ld_pos] : Zero4;
    input_data_cache[a_smem_st_offset] = v;
    
    bool in_image1 =  ld_h + 4 >= 0 && ld_h + 4 < params.input_height && w_in_image ;
    v = in_image1 ? in[ld_pos + 4 * params.input_width] : Zero4;
    input_data_cache[a_smem_st_offset +  32] = v;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (!any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch))) {
        auto z_out = out + (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
        auto result = params.has_bias ? biasTerms[gid.z] : Zero4;
        auto z_wt  = wt  + (int)gid.z * params.kernel_size;
        
        int offset_x = thread_index % 4;
        int y = thread_index / 4;
#pragma unroll
        for (auto kx = 0, x = offset_x; kx < 5; kx++, x ++) {
            auto wt4 = z_wt[kx];
            auto in4 = input_data_cache[ y * 8 + x];
            result += in4 * wt4;
        }
        
        *z_out = activate(result, params.activation);
    }
}

kernel void convolution_depthwise1x5_h4w8(const device ftype4 *in           [[buffer(0)]],
                                          device ftype4 *out                [[buffer(1)]],
                                          constant MetalConvParams& params  [[buffer(2)]],
                                          const device ftype4 *wt           [[buffer(3)]],
                                          const device ftype4 *biasTerms    [[buffer(4)]],
                                          uint3 gid                       [[thread_position_in_grid]],
                                          uint3 group_id                  [[threadgroup_position_in_grid]],
                                          uint thread_index               [[thread_index_in_threadgroup]]) {
    threadgroup ftype4 input_data_cache[8 * 8];
    
    // compute ld offset of inputs
    const int ld_start_w = group_id.x * 8 - params.pad_x;
    const int ld_start_h = group_id.y * 4 - params.pad_y;
    const int ld_start_c = group_id.z;
    
    const int ld_offset = ld_start_c * params.input_size;
    
    const int a_smem_st_offset = thread_index;
    
    // load data
    int ld_w = ld_start_w + thread_index % 8;
    int ld_h = ld_start_h + thread_index / 8;
    const int ld_pos = ld_offset + ld_h * params.input_width + ld_w;
    
    bool w_in_image = ld_w >=0 && ld_w < params.input_width;
    
    bool in_image = (ld_h >=0 && ld_h < params.input_height) && w_in_image;
    ftype4 v = in_image ? in[ld_pos] : Zero4;
    input_data_cache[a_smem_st_offset] = v;
    
    bool in_image1 =  ld_h + 4 >= 0 && ld_h + 4 < params.input_height && w_in_image ;
    v = in_image1 ? in[ld_pos + 4 * params.input_width] : Zero4;
    input_data_cache[a_smem_st_offset +  32] = v;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (!any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch))) {
        auto z_out = out + (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
        auto result = params.has_bias ? biasTerms[gid.z] : Zero4;
        auto z_wt  = wt  + (int)gid.z * params.kernel_size;
        
        int x = thread_index % 8;
        int offset_y = thread_index / 8;
#pragma unroll
        for (auto ky = 0, y = offset_y; ky < 5; ky++, y ++) {
            auto wt4 = z_wt[ky];
            auto in4 = input_data_cache[ y * 8 + x];
            result += in4 * wt4;
        }
        
        *z_out = activate(result, params.activation);
    }
}
