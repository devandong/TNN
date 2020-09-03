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

#include "test/unit_test/layer_test/layer_test.h"
#include "test/unit_test/unit_test_common.h"
#include "test/unit_test/utils/network_helpers.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

class ReshapeLayerTest : public LayerTest, public ::testing::WithParamInterface<std::tuple<int, int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, ReshapeLayerTest, ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE, testing::Values(0, 1)));

TEST_P(ReshapeLayerTest, ReshapeLayer) {
    // get param
    int batch        = std::get<0>(GetParam());
    int channel      = std::get<1>(GetParam());
    int input_size   = std::get<2>(GetParam());
    int reshape_type = std::get<3>(GetParam());
    DeviceType dev   = ConvertDeviceType(FLAGS_dt);

    batch = 1;
    channel = 6;
    int input_height = 2;
    int input_width = 4;

    // blob desc
    //auto inputs_desc  = CreateInputBlobsDesc(batch, channel, input_size, 1, DATA_TYPE_FLOAT);
    auto inputs_desc  = CreateInputBlobsDesc(batch, channel, input_height, input_width, 1, DATA_TYPE_FLOAT);
    auto outputs_desc = CreateOutputBlobsDesc(1, DATA_TYPE_FLOAT);

    // param
    ReshapeLayerParam param;
    param.name         = "Reshape";
    param.reshape_type = reshape_type;
    param.axis         = 0;
    param.num_axes     = 4;
    param.shape        = {0, 8, 3, 2};

    Run(LAYER_RESHAPE, &param, nullptr, inputs_desc, outputs_desc);
}

}  // namespace TNN_NS
