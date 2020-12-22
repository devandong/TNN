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

#ifndef TNN_EXAMPLES_BASE_POSE2D_DETECTOR_H_
#define TNN_EXAMPLES_BASE_POSE2D_DETECTOR_H_

#include "tnn_sdk_sample.h"
#include "landmark_smoothing_filter.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

namespace TNN_NS {

typedef ObjectInfo SkeletonInfo;

class SkeletonDetectorInput : public TNNSDKInput {
public:
    SkeletonDetectorInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~SkeletonDetectorInput(){}
};

/*
 the output of the efficient pose2d model is a list of 2d points:
 the output of the skeleton model is a list of 2d points:
 point0: nose
 point1: center between left shoulder and right shoulder
 point2: left  shoulder
 point3: left  elbow
 point4: left  wrist
 point5: right shoulder
 point6: right elbow
 point7: right wrist
 point8: left  hip
 point9: left  knee
 point10:left  ankle
 point11:right hip
 point12:right knee
 point13:right ankle
 point14:left eye
 point15:right eye
 point16: left  ear
 point17: right ear
 */

class SkeletonDetectorOutput : public TNNSDKOutput {
public:
    SkeletonDetectorOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~SkeletonDetectorOutput() {};
    SkeletonInfo keypoints;
    std::vector<float> confidence_list;
    std::vector<bool> detected;
};

class SkeletonDetectorOption : public TNNSDKOption {
public:
    SkeletonDetectorOption() {}
    virtual ~SkeletonDetectorOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
    float min_threshold = 0.15;
    int fps = 20;
};

class Pose2dDetector : public TNNSDKSample {
public:
    virtual ~Pose2dDetector() {};
    
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                            std::string name = kTNNSDKDefaultName);
    
private:
    void GenerateSkeleton(SkeletonDetectorOutput* output, std::shared_ptr<TNN_NS::Mat> heatmap,
                          float threshold);
    void SmoothingLandmarks(SkeletonDetectorOutput* output);
    void DeNormalize(SkeletonDetectorOutput* output);
    // the input mat size
    int orig_input_width;
    int orig_input_height;
    std::vector<SkeletonInfo> history;
    // lines for the efficient pose2d model
    std::vector<std::pair<int, int>> lines = {
        {0, 14},
        {0, 15},
        {1, 2},
        {1, 5},
        {2, 3},
        {2, 8},
        {3, 4},
        {5, 6},
        {5, 11},
        {6, 7},
        {8, 11},
        {8, 9},
        {9, 10},
        {11,12},
        {12,13},
        {14, 16},
        {15, 17}
    };
    // landmark filtering options
    const int window_size = 5;
    const float velocity_scale = 10.0;
    const float min_allowed_object_scale = 1e-6;
    std::shared_ptr<VelocityFilter> landmark_filter;
};

}

#endif // TNN_EXAMPLES_BASE_POSE2D_DETECTOR_H_


