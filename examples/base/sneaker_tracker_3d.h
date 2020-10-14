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

#ifndef TNN_EXAMPLES_BASE_SNEAKER_TRACKER_3D_H_
#define TNN_EXAMPLES_BASE_SNEAKER_TRACKER_3D_H_

#include "tnn_sdk_sample.h"
#include "tnn/utils/dims_vector_utils.h"
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <utility>

namespace TNN_NS {

class SneakerTracker3DInput : public TNNSDKInput {
public:
    SneakerTracker3DInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~SneakerTracker3DInput(){}
    std::vector<ObjectInfo> object_list;
};

class SneakerTracker3DOutput : public TNNSDKOutput {
public:
    SneakerTracker3DOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~SneakerTracker3DOutput();
};


class SneakerTracker3DOption : public TNNSDKOption {
public:
    SneakerTracker3DOption() {}
    virtual ~SneakerTracker3DOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
};

class SneakerTracker3D : public TNN_NS::TNNSDKSample {
public:
    ~SneakerTracker3D();
    virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);
private:
    ;
};
}
#endif // TNN_EXAMPLES_BASE_SNEAKER_TRACKER_3D_H_
