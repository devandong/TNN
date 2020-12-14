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

#define CONV_UNROLL (4)

kernel void convolution_1x1(const device ftype4 *in           [[buffer(0)]],
                            device ftype4 *out                [[buffer(1)]],
                            constant MetalConvParams& params  [[buffer(2)]],
                            const device ftype4x4 *wt         [[buffer(3)]],
                            const device ftype4 *biasTerms    [[buffer(4)]],
                            uint3 gid                         [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size,
                           params.output_slice_per_group,
                           params.batch)))
        return;
    
    int g = gid.y / params.output_slice_per_group;
    auto xy_wt  = wt                                                    + (int)gid.y * params.input_slice_per_group;
    auto xy_in  = in  + (int)gid.z * params.input_slice  * params.input_size  +          g * params.input_size  + (int)gid.x;
    auto xy_out = out + (int)gid.z * params.output_slice * params.output_size + (int)gid.y * params.output_size + (int)gid.x;
    
    auto result = params.has_bias ? float4(biasTerms[gid.y]) : float4(Zero4);
    for (auto z = 0; z < params.input_slice_per_group; z++, xy_in += params.input_size) {
        result += float4(*xy_in) * float4x4(xy_wt[z]);
    }
    *xy_out = activate(ftype4(result), params.activation);
}

#define USE_MATRIX 1

kernel void convolution_1x1_g1_h2w1(const device ftype4 *in           [[buffer(0)]],
                            device ftype4 *out                [[buffer(1)]],
                            constant MetalConvParams& params  [[buffer(2)]],
#if USE_MATRIX
                            const device ftype4x4 *wt         [[buffer(3)]],
#else
                            const device ftype4 *wt         [[buffer(3)]],
#endif
                            const device ftype4 *biasTerms    [[buffer(4)]],
                            uint3 gid                         [[thread_position_in_grid]]) {
    int ox = gid.x;
    int oy = gid.y * 2;
    if ( ox >= params.output_width || oy >= params.output_height)
        return;
    int os = gid.z % params.output_slice;
    int batch = gid.z / params.output_slice;
    
    float4 r00 = params.has_bias? float4(biasTerms[os]) : float4(Zero4);
    float4 r10 = params.has_bias? float4(biasTerms[os]) : float4(Zero4);
    
    
#if USE_MATRIX
    const device ftype4x4* weights = wt + os*params.input_slice;
#else
    const device ftype4* weights = wt + os*params.input_slice*4;
#endif
    
    int iy0 = clamp(oy+0, 0, params.input_height-1);
    int iy1 = clamp(oy+1, 0, params.input_height-1);
    int ix  = ox;
    const device ftype4* src00_loc = in + batch * params.input_slice * params.input_size + iy0 * params.input_width + ix;
    const device ftype4* src10_loc = in + batch * params.input_slice * params.input_size + iy1 * params.input_width + ix;
    
    for(int s=0; s<params.input_slice; ++s) {
        ftype4 src00 = *src00_loc;
        ftype4 src10 = *src10_loc;
#if USE_MATRIX
        r00 += float4(ftype4(src00) * ftype4x4(weights[0]));
        r10 += float4(ftype4(src10) * ftype4x4(weights[0]));
        weights += 1;
#else
        r00.x += dot(src00, weights[0]);
        r10.x += dot(src10, weights[0]);
        
        r00.y += dot(src00, weights[1]);
        r10.y += dot(src10, weights[1]);
        
        r00.z += dot(src00, weights[2]);
        r10.z += dot(src10, weights[2]);
        
        r00.w += dot(src00, weights[3]);
        r10.w += dot(src10, weights[3]);
        weights += 4;
#endif
        
        src00_loc += params.input_size;
        src10_loc += params.input_size;
        
    }
    
    device ftype4* out00_loc = out + (batch * params.output_slice + os) * params.output_size + oy * params.output_width + ox;
    device ftype4* out10_loc = out00_loc + params.output_width;
    
    *out00_loc = activate(ftype4(r00), params.activation);
    if (oy + 1 < params.output_height)
        *out10_loc = activate(ftype4(r10), params.activation);
}

kernel void convolution_1x1_g1_c2h2w1(const device ftype4 *in           [[buffer(0)]],
                            device ftype4 *out                [[buffer(1)]],
                            constant MetalConvParams& params  [[buffer(2)]],
                            const device ftype4 *wt         [[buffer(3)]],
                            const device ftype4 *biasTerms    [[buffer(4)]],
                            uint3 gid                         [[thread_position_in_grid]]) {
    int ox = gid.x;
    int oy = gid.y * 2;
    int os = gid.z * 2;
    if ( ox >= params.output_width || oy >= params.output_height || os >= params.output_slice)
        return;
    const int batch = 0;
    
    float4 r000 = params.has_bias? float4(biasTerms[os]) : float4(Zero4);
    float4 r010 = params.has_bias? float4(biasTerms[os]) : float4(Zero4);
    float4 r100 = params.has_bias? float4(biasTerms[os+1]) : float4(Zero4);
    float4 r110 = params.has_bias? float4(biasTerms[os+1]) : float4(Zero4);
    
    const device ftype4* weights0 = wt + (os+0)*params.input_slice*4;
    const device ftype4* weights1 = wt + (os+1)*params.input_slice*4;
    
    int iy0 = clamp(oy+0, 0, params.input_height-1);
    int iy1 = clamp(oy+1, 0, params.input_height-1);
    int ix  = ox;
    const device ftype4* src00_loc = in + batch * params.input_slice * params.input_size + iy0 * params.input_width + ix;
    const device ftype4* src10_loc = in + batch * params.input_slice * params.input_size + iy1 * params.input_width + ix;
    
    for(int s=0; s<params.input_slice; ++s) {
        ftype4 src00 = *src00_loc;
        ftype4 src10 = *src10_loc;
        
        r000.x += dot(src00, weights0[0]);
        r010.x += dot(src10, weights0[0]);
        
        r000.y += dot(src00, weights0[1]);
        r010.y += dot(src10, weights0[1]);
        
        r000.z += dot(src00, weights0[2]);
        r010.z += dot(src10, weights0[2]);
        
        r000.w += dot(src00, weights0[3]);
        r010.w += dot(src10, weights0[3]);
        
        weights0 += 4;
        
        r100.x += dot(src00, weights1[0]);
        r110.x += dot(src10, weights1[0]);
        
        r100.y += dot(src00, weights1[1]);
        r110.y += dot(src10, weights1[1]);
        
        r100.z += dot(src00, weights1[2]);
        r110.z += dot(src10, weights1[2]);
        
        r100.w += dot(src00, weights1[3]);
        r110.w += dot(src10, weights1[3]);
        
        weights1 += 4;
        
        src00_loc += params.input_size;
        src10_loc += params.input_size;
        
    }
    
    device ftype4* out000_loc = out + (batch * params.output_slice + os) * params.output_size + oy * params.output_width + ox;
    device ftype4* out010_loc = out000_loc + params.output_width;
    device ftype4* out100_loc = out000_loc + params.output_size;
    device ftype4* out110_loc = out100_loc + params.output_width;
    
    *out000_loc = activate(ftype4(r000), params.activation);
    if (oy + 1 < params.output_height)
        *out010_loc = activate(ftype4(r010), params.activation);
    
    if (os + 1 < params.output_slice) {
        *out100_loc = activate(ftype4(r100), params.activation);
        if (oy + 1 < params.output_height)
            *out110_loc = activate(ftype4(r110), params.activation);
    }
}

kernel void convolution_1x1_g1_c4h2w1(const device ftype4 *in           [[buffer(0)]],
                            device ftype4 *out                [[buffer(1)]],
                            constant MetalConvParams& params  [[buffer(2)]],
                            const device ftype4 *wt         [[buffer(3)]],
                            const device ftype4 *biasTerms    [[buffer(4)]],
                            uint3 gid                         [[thread_position_in_grid]]) {
    int ox = gid.x;
    int oy = gid.y * 2;
    int os = gid.z * 4;
    if ( ox >= params.output_width || oy >= params.output_height || os >= params.output_slice)
        return;
    const int batch = 0;
    
    float4 r000 = params.has_bias? float4(biasTerms[os]) : float4(Zero4);
    float4 r010 = params.has_bias? float4(biasTerms[os]) : float4(Zero4);
    float4 r100 = params.has_bias? float4(biasTerms[os+1]) : float4(Zero4);
    float4 r110 = params.has_bias? float4(biasTerms[os+1]) : float4(Zero4);
    float4 r200 = params.has_bias? float4(biasTerms[os+2]) : float4(Zero4);
    float4 r210 = params.has_bias? float4(biasTerms[os+2]) : float4(Zero4);
    float4 r300 = params.has_bias? float4(biasTerms[os+3]) : float4(Zero4);
    float4 r310 = params.has_bias? float4(biasTerms[os+3]) : float4(Zero4);
    
    const device ftype4* weights0 = wt + (os+0)*params.input_slice*4;
    const device ftype4* weights1 = wt + (os+1)*params.input_slice*4;
    const device ftype4* weights2 = wt + (os+2)*params.input_slice*4;
    const device ftype4* weights3 = wt + (os+3)*params.input_slice*4;
    
    int iy0 = clamp(oy+0, 0, params.input_height-1);
    int iy1 = clamp(oy+1, 0, params.input_height-1);
    int ix  = ox;
    const device ftype4* src00_loc = in + batch * params.input_slice * params.input_size + iy0 * params.input_width + ix;
    const device ftype4* src10_loc = in + batch * params.input_slice * params.input_size + iy1 * params.input_width + ix;
    
    for(int s=0; s<params.input_slice; ++s) {
        ftype4 src00 = *src00_loc;
        ftype4 src10 = *src10_loc;
        
        r000.x += dot(src00, weights0[0]);
        r010.x += dot(src10, weights0[0]);
        r000.y += dot(src00, weights0[1]);
        r010.y += dot(src10, weights0[1]);
        r000.z += dot(src00, weights0[2]);
        r010.z += dot(src10, weights0[2]);
        r000.w += dot(src00, weights0[3]);
        r010.w += dot(src10, weights0[3]);
        
        weights0 += 4;
        
        r100.x += dot(src00, weights1[0]);
        r110.x += dot(src10, weights1[0]);
        r100.y += dot(src00, weights1[1]);
        r110.y += dot(src10, weights1[1]);
        r100.z += dot(src00, weights1[2]);
        r110.z += dot(src10, weights1[2]);
        r100.w += dot(src00, weights1[3]);
        r110.w += dot(src10, weights1[3]);
        
        weights1 += 4;
        
        r200.x += dot(src00, weights2[0]);
        r210.x += dot(src10, weights2[0]);
        r200.y += dot(src00, weights2[1]);
        r210.y += dot(src10, weights2[1]);
        r200.z += dot(src00, weights2[2]);
        r210.z += dot(src10, weights2[2]);
        r200.w += dot(src00, weights2[3]);
        r210.w += dot(src10, weights2[3]);
        
        weights2 += 4;
        
        r300.x += dot(src00, weights3[0]);
        r310.x += dot(src10, weights3[0]);
        r300.y += dot(src00, weights3[1]);
        r310.y += dot(src10, weights3[1]);
        r300.z += dot(src00, weights3[2]);
        r310.z += dot(src10, weights3[2]);
        r300.w += dot(src00, weights3[3]);
        r310.w += dot(src10, weights3[3]);
        
        weights3 += 4;
        
        src00_loc += params.input_size;
        src10_loc += params.input_size;
        
    }
    
    device ftype4* out000_loc = out + (batch * params.output_slice + os) * params.output_size + oy * params.output_width + ox;
    device ftype4* out010_loc = out000_loc + params.output_width;
    device ftype4* out100_loc = out000_loc + params.output_size;
    device ftype4* out110_loc = out100_loc + params.output_width;
    device ftype4* out200_loc = out100_loc + params.output_size;
    device ftype4* out210_loc = out200_loc + params.output_width;
    device ftype4* out300_loc = out200_loc + params.output_size;
    device ftype4* out310_loc = out300_loc + params.output_width;
    
    *out000_loc = activate(ftype4(r000), params.activation);
    if (oy + 1 < params.output_height)
        *out010_loc = activate(ftype4(r010), params.activation);
    
    if (os + 1 < params.output_slice) {
        *out100_loc = activate(ftype4(r100), params.activation);
        if (oy + 1 < params.output_height)
            *out110_loc = activate(ftype4(r110), params.activation);
    }
    
    if (os + 2 < params.output_slice) {
        *out200_loc = activate(ftype4(r200), params.activation);
        if (oy + 1 < params.output_height)
            *out210_loc = activate(ftype4(r210), params.activation);
    }
    
    if (os + 3 < params.output_slice) {
        *out300_loc = activate(ftype4(r300), params.activation);
        if (oy + 1 < params.output_height)
            *out310_loc = activate(ftype4(r310), params.activation);
    }
}


kernel void convolution_1x1_g1z4(const device ftype4 *in             [[buffer(0)]],
                                 device ftype4 *out                  [[buffer(1)]],
                                 constant MetalConvParams& params    [[buffer(2)]],
                                 const device ftype4x4 *wt           [[buffer(3)]],
                                 const device ftype4 *biasTerms      [[buffer(4)]],
                                 uint3 gid                           [[thread_position_in_grid]]) {
    if ((int)gid.x >= params.output_size || (int)gid.y * CONV_UNROLL >= params.output_slice || (int)gid.z >= params.batch) return;
    
    int uz = gid.y * CONV_UNROLL;
    auto xy_wt0 = wt + uz * params.input_slice;
    auto xy_wt1 = uz + 1 < params.output_slice ? xy_wt0 + params.input_slice : nullptr;
    auto xy_wt2 = uz + 2 < params.output_slice ? xy_wt1 + params.input_slice : nullptr;
    auto xy_wt3 = uz + 3 < params.output_slice ? xy_wt2 + params.input_slice : nullptr;
    auto xy_in  = in  + (int)gid.z * params.input_slice  * params.input_size                         + (int)gid.x;
    auto xy_out = out + (int)gid.z * params.output_slice * params.output_size + uz * params.output_size + (int)gid.x;
    
    float4 result0 = 0, result1 = 0, result2 = 0, result3 = 0;
    for (auto z = 0; z < params.input_slice; z++, xy_in += params.input_size) {
        auto in4 = float4(*xy_in);
        /* true */  result0 += in4 * float4x4(xy_wt0[z]);
        if (xy_wt1) result1 += in4 * float4x4(xy_wt1[z]);
        if (xy_wt2) result2 += in4 * float4x4(xy_wt2[z]);
        if (xy_wt3) result3 += in4 * float4x4(xy_wt3[z]);
    }
    
    
    if (params.has_bias) {
        *xy_out = activate(ftype4(result0 + float4(biasTerms[uz + 0])), params.activation);
        if (xy_wt1) { xy_out += params.output_size; *xy_out = activate(ftype4(result1 + float4(biasTerms[uz + 2])), params.activation); }
        if (xy_wt2) { xy_out += params.output_size; *xy_out = activate(ftype4(result2 + float4(biasTerms[uz + 2])), params.activation); }
        if (xy_wt3) { xy_out += params.output_size; *xy_out = activate(ftype4(result3 + float4(biasTerms[uz + 3])), params.activation); }
    } else {
        *xy_out = activate(ftype4(result0), params.activation);
        if (xy_wt1) { xy_out += params.output_size; *xy_out = activate(ftype4(result1), params.activation); }
        if (xy_wt2) { xy_out += params.output_size; *xy_out = activate(ftype4(result2), params.activation); }
        if (xy_wt3) { xy_out += params.output_size; *xy_out = activate(ftype4(result3), params.activation); }
    }
}

