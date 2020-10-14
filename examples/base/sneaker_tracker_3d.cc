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

#include "sneaker_tracker_3d.h"

namespace TNN_NS {

void ComputeBoundingRect(ObjectInfo& object) {
    float top = 1.0f;
    float bottom = 0.0f;
    float left = 1.0f;
    float right = 0.0f;
    for (const auto& point : object.key_points) {
        top = std::min(top, point.second);
        bottom = std::max(bottom, point.second);
        left = std::min(left, point.first);
        right = std::max(right, point.first);
    }
    object.x1 = right;
    object.y1 = bottom;
    object.x2 = left;
    object.y2 = top;
    //TODO: check if the id is used in mediapipe
    object.class_id = 0;
}

Status SneakerTracker3D::Predict(std::shared_ptr<TNNSDKInput> input_, std::shared_ptr<TNNSDKOutput> &output) {
    SneakerTracker3DInput* input = dynamic_cast<SneakerTracker3DInput *>(input_.get());
    RETURN_VALUE_ON_NEQ(input==nullptr, false, Status(TNNERR_PARAM_ERR, "invalid input!"));

    // 1) compute box
    auto objects = input->object_list;
    for (auto& object : objects) {
        ComputeBoundingRect(object);
    }
    
    // 2) resize
    auto input_mat = input->GetMat();
    auto input_dims = input_mat->GetDims();
    int input_height = input_dims[2];
    int input_width  = input_dims[3];
    
    std::vector<int> target_dims = {1, 4, 320, 240};
    int target_height = target_dims[2];
    int target_width  = target_dims[3];
    
    int output_width  = target_width;
    int output_height = target_height;

    auto resized_input_mat = input_mat;
    if (input_height != target_height || input_width != target_width) {
        const float scale = std::min(static_cast<float>(target_width) / input_width,
                                     static_cast<float>(target_height) / input_height);
        const int resized_width  = std::round(input_width * scale);
        const int resized_height = std::round(input_height * scale);
        // TODO: we should use INTER_AREA when scale<1.0, use INTER_LINEAR for now, as TNN does not support INTER_AREA
        auto scale_flag = scale < 1.0f ? TNNInterpLinear : TNNInterpLinear;
        
        auto resized_dims = input_dims;
        resized_dims[2] = resized_height;
        resized_dims[3] = resized_width;
        resized_input_mat = std::make_shared<Mat>(DEVICE_ARM, N8UC4, resized_dims);
        auto status = Resize(input_mat, resized_input_mat, scale_flag);
        RETURN_ON_NEQ(status, TNN_OK);

        output_width  = resized_width;
        output_height = resized_height;
    }

    return TNN_OK;
}

}
