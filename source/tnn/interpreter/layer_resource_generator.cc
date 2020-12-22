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

#include "tnn/interpreter/layer_resource_generator.h"

#include <mutex>

namespace TNN_NS {

std::map<LayerType, std::shared_ptr<LayerResourceGenerator>>& GetGlobalLayerResourceGeneratorMap() {
    static std::once_flag once;
    static std::shared_ptr<std::map<LayerType, std::shared_ptr<LayerResourceGenerator>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<LayerType, std::shared_ptr<LayerResourceGenerator>>); });
    return *creators;
}

Status GenerateRandomResource(LayerType type, LayerParam* param, LayerResource** resource, std::vector<Blob*>& inputs) {
    auto& layer_resource_generator_map = GetGlobalLayerResourceGeneratorMap();
    if (layer_resource_generator_map.count(type) > 0) {
        auto layer_name = param->name;
        printf("===== Generate random resource for:%s\n", layer_name.c_str());
        auto generator = layer_resource_generator_map[type]->GenLayerResource(param, resource, inputs);
    }
    return TNN_OK;
}

/*
 * Fill with specific generator
 */
template <typename T>
void GaussianFill1D(T*ptr, int ksize, float sigma) {
    if (sigma <= 0)
        return;
    if (ksize %2 != 1)
        return;

    const double sd_minus_0_125 = -0.125;
    double scale_2x = sd_minus_0_125 / (sigma * sigma);
    
    int length_half = (ksize - 1) / 2;
    std::vector<double> values(length_half +  1);
    double sum = 0;
    for(int i=0, x=1-ksize; i<length_half; ++i, x+=2) {
        double t = exp(static_cast<double>(x*x) * scale_2x);
        values[i] = t;
        sum += t;
    }
    sum *= 2;
    sum += 1;
    
    double mul1 = static_cast<double>(1.0) / sum;
    for(int i=0; i<length_half; ++i) {
        double t = values[i] * mul1;
        ptr[i] = t;
        ptr[ksize - 1 - i] = t;
    }
    ptr[length_half] = 1 * mul1;
}

template <typename T>
void GaussianFill1D(T*ptr, int count, int ksize, float sigma) {
    printf("===== Generate Guissian 1D  weights, %d planes, %d floats in a kernel\n",
            count, ksize);
    for(int c=0; c<count; ++c) {
        GaussianFill1D(ptr+c*ksize, ksize, sigma);
    }
}

template <typename T>
void GaussianFill2D(T*ptr, int count, int kh, int kw, float sigma_w, float sigma_h) {
     printf("===== Generate Guissian 2D  weights, %d planes, %d floats in a kernel\n",
            count, kh*kw);
    T* buffer = new T[kh+kw];
    GaussianFill1D(buffer, kw, sigma_w);
    GaussianFill1D(buffer+kw, kh, sigma_h);
    for (int n = 0; n < count; ++n) {
        T* ptr_c = ptr + n*kh*kw;
        for (int h = 0; h < kh; ++h) {
            for (int w = 0; w < kw; ++w) {
                ptr_c[h * kw + w] = buffer[w] * buffer[kw + h];
            }
        }
    }
    delete[] buffer;
}

/*
 * Generate conv resource
 */
class ConvolutionLayerResourceGenerator : public LayerResourceGenerator {
    virtual Status GenLayerResource(LayerParam* param, LayerResource** resource, std::vector<Blob*>& inputs) {
        LOGD("ConvolutionLayerResourceGenerator\n");
        auto layer_param = dynamic_cast<ConvLayerParam*>(param);
        CHECK_PARAM_NULL(layer_param);
        // auto layer_res   = dynamic_cast<ConvLayerResource*>(resource);
        auto layer_res = new ConvLayerResource();

        printf("===== Generate gassian weights for layer:%s\n", param->name.c_str());
        auto dims = inputs[0]->GetBlobDesc().dims;
        if (layer_param->quantized) {
            layer_res->filter_handle = RawBuffer(dims[1] * layer_param->output_channel * layer_param->kernels[0] *
                                                 layer_param->kernels[1] / layer_param->group * sizeof(int8_t));
            layer_res->bias_handle   = RawBuffer(layer_param->output_channel * sizeof(int32_t));
            layer_res->scale_handle  = RawBuffer(layer_param->output_channel * sizeof(float));
            layer_res->filter_handle.SetDataType(DATA_TYPE_INT8);
            layer_res->bias_handle.SetDataType(DATA_TYPE_INT32);
            layer_res->scale_handle.SetDataType(DATA_TYPE_FLOAT);
        } else {
            layer_res->filter_handle =
                RawBuffer(dims[1]* layer_param->output_channel * layer_param->kernels[0] *
                          layer_param->kernels[1] / layer_param->group * sizeof(float));
                if (layer_param->kernels[0] == 1 ) {
                    GaussianFill1D(layer_res->filter_handle.force_to<float*>(),
                                    dims[1]* layer_param->output_channel/layer_param->group,
                                    layer_param->kernels[1], 3.0);
                } else if (layer_param->kernels[1] == 1 ) {
                    GaussianFill1D(layer_res->filter_handle.force_to<float*>(),
                                    dims[1]* layer_param->output_channel/layer_param->group,
                                    layer_param->kernels[0], 3.0);
                } else {
                    GaussianFill2D(layer_res->filter_handle.force_to<float*>(),
                                    dims[1]* layer_param->output_channel/layer_param->group,
                                    layer_param->kernels[1], 
                                    layer_param->kernels[0], 3.0, 3.0);
                }

            if (layer_param->bias) {
                layer_res->bias_handle = RawBuffer(layer_param->output_channel * sizeof(float));
            }
        }

        *resource = layer_res;
        return TNN_OK;
    }
};

/*
 * Generate deconv resource
 */
class DeconvolutionLayerResourceGenerator : public ConvolutionLayerResourceGenerator {};

/*
 * Generate weights for innerproduct layer
 */
class InnerProductLayerResourceGenerator : public LayerResourceGenerator {
    virtual Status GenLayerResource(LayerParam* param, LayerResource** resource, std::vector<Blob*>& inputs) {
        LOGD("InnerProductLayerResourceGenerator\n");
        auto layer_param = dynamic_cast<InnerProductLayerParam*>(param);
        CHECK_PARAM_NULL(layer_param);
        auto layer_res = new InnerProductLayerResource();

        auto dims = inputs[0]->GetBlobDesc().dims;

        if (param->quantized) {
            layer_res->weight_handle =
                RawBuffer(layer_param->num_output * dims[1] * dims[2] * dims[3] * sizeof(int8_t));
            layer_res->bias_handle  = RawBuffer(layer_param->num_output * sizeof(int32_t));
            layer_res->scale_handle = RawBuffer(layer_param->num_output * sizeof(float));
            layer_res->weight_handle.SetDataType(DATA_TYPE_INT8);
            layer_res->bias_handle.SetDataType(DATA_TYPE_INT32);
            layer_res->scale_handle.SetDataType(DATA_TYPE_FLOAT);
        } else {
            layer_res->weight_handle = RawBuffer(layer_param->num_output * dims[1] * dims[2] * dims[3] * sizeof(float));

            if (layer_param->has_bias) {
                layer_res->bias_handle = RawBuffer(layer_param->num_output * sizeof(float));
            }
        }

        *resource = layer_res;
        return TNN_OK;
    }
};

/*
 * Generate weights for Batchnorm layer
 */
class BatchnormLayerResourceGenerator : public LayerResourceGenerator {
    virtual Status GenLayerResource(LayerParam* param, LayerResource** resource, std::vector<Blob*>& inputs) {
        LOGD("BatchnormLayerResourceGenerator\n");
        auto layer_res = new BatchNormLayerResource();

        auto dims = inputs[0]->GetBlobDesc().dims;

        layer_res->scale_handle = RawBuffer(dims[1] * sizeof(float));
        layer_res->bias_handle  = RawBuffer(dims[1] * sizeof(float));

        *resource = layer_res;
        return TNN_OK;
    }
};

/*
 * Generate weights for InstanceNorm layer
 */
class InstanceNormLayerResourceGenerator : public LayerResourceGenerator {
    virtual Status GenLayerResource(LayerParam* param, LayerResource** resource, std::vector<Blob*>& inputs) {
        LOGD("InstanceNormLayerResourceGenerator\n");
        auto layer_res = new InstanceNormLayerResource();

        auto dims = inputs[0]->GetBlobDesc().dims;

        layer_res->scale_handle = RawBuffer(dims[1] * sizeof(float));
        layer_res->bias_handle  = RawBuffer(dims[1] * sizeof(float));

        *resource = layer_res;
        return TNN_OK;
    }
};

/*
 * Generate weights for Prelu layer
 */
class PReluLayerResourceGenerator : public LayerResourceGenerator {
    virtual Status GenLayerResource(LayerParam* param, LayerResource** resource, std::vector<Blob*>& inputs) {
        LOGD("PReluLayerResourceGenerator\n");
        auto layer_res = new PReluLayerResource();

        auto dims = inputs[0]->GetBlobDesc().dims;

        layer_res->slope_handle = RawBuffer(dims[1] * sizeof(float));

        *resource = layer_res;
        return TNN_OK;
    }
};

/*
 * Generate weights for Blobscale
 */
class BlobScaleLayerResourceGenerator : public LayerResourceGenerator {
    virtual Status GenLayerResource(LayerParam* param, LayerResource** resource, std::vector<Blob*>& inputs) {
        LOGD("BlobScaleLayerResourceGenerator\n");
        auto layer_res = new IntScaleResource();

        auto dims = inputs[0]->GetBlobDesc().dims;

        layer_res->scale_handle = RawBuffer(dims[1] * sizeof(float));
        layer_res->bias_handle  = RawBuffer(dims[1] * sizeof(int32_t));
        layer_res->scale_handle.SetDataType(DATA_TYPE_FLOAT);
        layer_res->bias_handle.SetDataType(DATA_TYPE_INT32);

        *resource = layer_res;
        return TNN_OK;
    }
};

/*
 * Generate weights for Binary
 */
class BinaryLayerResourceGenerator : public LayerResourceGenerator {
    virtual Status GenLayerResource(LayerParam* param, LayerResource** resource, std::vector<Blob*>& inputs) {
        LOGD("BinaryLayerResourceGenerator\n");

        if (inputs.size() == 1) {
            LOGE(
                "[WARNNING] can't infer resource shape from binary param in benchmark mode, random generator may not "
                "be exactly same with the real resource!\n");
            auto layer_res           = new EltwiseLayerResource();
            auto dims                = inputs[0]->GetBlobDesc().dims;
            layer_res->element_shape = {1, 1, 1, 1};
            // broad cast in channel
            layer_res->element_shape[1] = dims[1];
            layer_res->element_handle   = RawBuffer(dims[1] * sizeof(float));

            *resource = layer_res;
        }

        return TNN_OK;
    }
};

class AddLayerResourceGenerator : public BinaryLayerResourceGenerator {};
class SubLayerResourceGenerator : public BinaryLayerResourceGenerator {};
class MaxLayerResourceGenerator : public BinaryLayerResourceGenerator {};
class MinLayerResourceGenerator : public BinaryLayerResourceGenerator {};
class DivLayerResourceGenerator : public BinaryLayerResourceGenerator {};
class MulLayerResourceGenerator : public BinaryLayerResourceGenerator {};

/*
 * Generate Hdr resource
 */
class HdrGuideLayerResourceGenerator : public LayerResourceGenerator {
    virtual Status GenLayerResource(LayerParam* param, LayerResource** resource, std::vector<Blob*>& inputs) {
        LOGD("HdrGuideLayerResourceGenerator\n");
        auto layer_res = new HdrGuideLayerResource();

        layer_res->ccm_weight_handle        = RawBuffer(9 * sizeof(float));
        layer_res->ccm_bias_handle          = RawBuffer(3 * sizeof(float));
        layer_res->shifts_handle            = RawBuffer(12 * sizeof(float));
        layer_res->slopes_handle            = RawBuffer(12 * sizeof(float));
        layer_res->projection_weight_handle = RawBuffer(3 * sizeof(float));
        layer_res->projection_bias_handle   = RawBuffer(1 * sizeof(float));

        *resource = layer_res;

        return TNN_OK;
    }
};

REGISTER_LAYER_RESOURCE(Convolution, LAYER_CONVOLUTION)
REGISTER_LAYER_RESOURCE(Deconvolution, LAYER_DECONVOLUTION)
REGISTER_LAYER_RESOURCE(InnerProduct, LAYER_INNER_PRODUCT)
REGISTER_LAYER_RESOURCE(Batchnorm, LAYER_BATCH_NORM)
REGISTER_LAYER_RESOURCE(InstanceNorm, LAYER_INST_BATCH_NORM)
REGISTER_LAYER_RESOURCE(PRelu, LAYER_PRELU)
REGISTER_LAYER_RESOURCE(BlobScale, LAYER_BLOB_SCALE)
REGISTER_LAYER_RESOURCE(Add, LAYER_ADD);
REGISTER_LAYER_RESOURCE(Sub, LAYER_SUB);
REGISTER_LAYER_RESOURCE(Max, LAYER_MAXIMUM);
REGISTER_LAYER_RESOURCE(Min, LAYER_MINIMUM);
REGISTER_LAYER_RESOURCE(Div, LAYER_DIV);
REGISTER_LAYER_RESOURCE(Mul, LAYER_MUL);
REGISTER_LAYER_RESOURCE(HdrGuide, LAYER_HDRGUIDE);
}  // namespace TNN_NS
