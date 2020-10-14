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
#define USE_EIGEN 1

#include "sneaker_detector_3d.h"

#ifdef USE_EIGEN
#include <Eigen/Eigen>
#endif

namespace TNN_NS {
struct BeliefBox{
    float belief;
    std::vector<std::pair<float, float>> box_2d;
};

//TODO: check if this aligns with opencv
// opencv::Dilate when MORPH_RECT && anchor=-1
std::shared_ptr<Mat> DilateRect(std::shared_ptr<Mat> src, const int size) {
    std::shared_ptr<Mat> dst = nullptr;
    if (src->GetMatType() != NCHW_FLOAT)
        return dst;
    
    auto dims = src->GetDims();
    dst = std::make_shared<Mat>(DEVICE_ARM, NCHW_FLOAT, dims);
    auto src_data = static_cast<float *>(src->GetData());
    auto dst_data = static_cast<float *>(dst->GetData());
    
    for(int h=0; h<dims[2]; ++h) {
        for(int w=0; w<dims[3]; ++w) {
            float val = -FLT_MAX;
            int start_src_h = h - size / 2;
            int start_src_w = w - size / 2;
            for(int kh=0; kh<size; ++kh) {
                int src_h = std::min(std::max(start_src_h + kh, 0), dims[2]);
                for(int kw=0; kw<size; ++kw) {
                    int src_w = std::min(std::max(start_src_w + kw, 0), dims[3]);
                    val = std::max(val, src_data[src_h * dims[3] + src_w]);
                }
            }
            dst_data[h * dims[3] + w] = val;
        }
    }
    return dst;
}

bool IsBelifBoxIdentical(const BeliefBox& box_1,
                          const BeliefBox& box_2, int voting_allowance) {
  // Skip the center point.
  for (int i = 1; i < box_1.box_2d.size(); ++i) {
    const float x_diff =
        std::abs(box_1.box_2d[i].first - box_2.box_2d[i].first);
    const float y_diff =
        std::abs(box_1.box_2d[i].second - box_2.box_2d[i].second);
    if (x_diff > voting_allowance || y_diff > voting_allowance) {
      return false;
    }
  }
  return true;
}

bool IsNewBeliefBox(std::vector<BeliefBox>* boxes, BeliefBox* box, int voting_allowance) {
  for (auto& b : *boxes) {
    if (IsBelifBoxIdentical(b, *box, voting_allowance)) {
      if (b.belief < box->belief) {
        std::swap(b, *box);
      }
      return false;
    }
  }
  return true;
}

void ExtractCenterKeypoints(std::shared_ptr<Mat> heatmap, std::vector<std::pair<int, int>>& locations) {
    locations.clear();

    constexpr int local_max_distance = 2;
    const int kernel_size = static_cast<int>(local_max_distance * 2 + 1 + 0.5f);
    auto max_filtered_heapmap = DilateRect(heatmap, kernel_size);
    
    // bitwise and
    constexpr float heatmap_threshold = 0.6;
    auto count = DimsVectorUtils::Count(heatmap->GetDims());

    const int heatmap_width = heatmap->GetWidth();
    auto heatmap_data = static_cast<float *>(heatmap->GetData());
    auto max_filtered_heatmap_data = static_cast<float *>(max_filtered_heapmap->GetData());
    for(int i=0; i<count; ++i) {
        if(heatmap_data[i] >= max_filtered_heatmap_data[i] && heatmap_data[i] >= heatmap_threshold) {
            locations.push_back(std::make_pair(i % heatmap_width, i / heatmap_width));
        }
    }
}

void DecodeByVoting(std::shared_ptr<Mat> heatmap, std::shared_ptr<Mat> offsetmap,
                    int center_x, int center_y, float offset_scale_x,
                    float offset_scale_y, int voting_radius, BeliefBox* box) {
    const int offsetmap_channel_num = offsetmap->GetChannel();
    auto heatmap_height = heatmap->GetHeight();
    auto heatmap_width  = heatmap->GetWidth();
    auto offsetmap_height = offsetmap->GetHeight();
    auto offsetmap_width  = offsetmap->GetWidth();
    auto offsetmap_hw = offsetmap_height * offsetmap_width;

    float* offsetmap_data = static_cast<float *>(offsetmap->GetData());
    float* heatmap_data   = static_cast<float *>(heatmap->GetData());

    std::vector<float> center_votes(offsetmap_channel_num, 0);
    auto center_offsetmap_offset = center_y * offsetmap_width + center_x;
    for(int i=0; i<offsetmap_channel_num/2; ++i) {
        auto channel_offset = 2 * i * offsetmap_hw;
        center_votes[2 * i]     = center_x +
                        offsetmap_data[channel_offset + center_offsetmap_offset] * offset_scale_x;
        channel_offset += offsetmap_hw;
        center_votes[2 * i + 1] = center_y +
                        offsetmap_data[channel_offset + center_offsetmap_offset] * offset_scale_y;
    }
    
    // Find voting window.
    int x_min  = std::max(0, center_x - voting_radius);
    int y_min  = std::max(0, center_y - voting_radius);
    int width  = std::min(heatmap_width  - x_min, voting_radius * 2 + 1);
    int height = std::min(heatmap_height - y_min, voting_radius * 2 + 1);

    // move to starting location
    offsetmap_data += y_min * offsetmap_width + x_min;
    heatmap_data   += y_min * heatmap_width   + x_min;
    constexpr float voting_threshold = 0.2;
    
    const int voting_allowance = 1;
    for(int i=0; i<offsetmap_channel_num/2; ++i) {
        float x_sum = 0.f;
        float y_sum = 0.f;
        float votes = 0.f;
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                const float belief = heatmap_data[r * heatmap_width + c];
                if (belief < voting_threshold) {
                    continue;
                }
                auto channel_offset = 2 * i * offsetmap_hw;
                auto spatial_offset = r * offsetmap_width + c;
                float offset_x = offsetmap_data[channel_offset + spatial_offset] * offset_scale_x;
                channel_offset += offsetmap_hw;
                float offset_y = offsetmap_data[channel_offset + spatial_offset] * offset_scale_y;
                float vote_x = c + x_min + offset_x;
                float vote_y = r + y_min + offset_y;
                float x_diff = std::abs(vote_x - center_votes[2 * i]);
                float y_diff = std::abs(vote_y - center_votes[2 * i + 1]);
                if (x_diff > voting_allowance || y_diff > voting_allowance) {
                    continue;
                }
                x_sum += vote_x * belief;
                y_sum += vote_y * belief;
                votes += belief;
            }
        }
        box->box_2d.emplace_back(x_sum / votes, y_sum / votes);
    }
}

void SneakerDetector3D::FitBox(const std::vector<Vector<float, 3>>& vertices) {
    if (vertices.size() != num_keypoints)
        return;
    box_scale.SetZero();
    for (int axis = 0; axis < num_axis; ++axis) {
        for (int edge_id = 0; edge_id < num_edges_per_axis; ++edge_id) {
            const std::array<int, 2>& edge = box_edges[axis * num_edges_per_axis + edge_id];
            //box_scale[axis] += (vertices[edge[0]] - vertices[edge[1]]).norm();
            auto rst = (vertices[edge[0]] - vertices[edge[1]]);
            box_scale[axis] += rst.norm();
        }
        box_scale[axis] /= num_edges_per_axis;
    }
    // Create a scaled axis-aligned box
    //transformation_.setIdentity();
    box_transformation.SetZero();
    box_transformation(0, 0) = 1;
    box_transformation(1, 1) = 1;
    box_transformation(2, 2) = 1;
    box_transformation(3, 3) = 1;
    
    //Update();
    UpdateBox();
    
    // MatrixN3_RM, 9*3 float matrix, row-major
    //using s = Eigen::Matrix<float, kNumKeypoints, 3, Eigen::RowMajor>;
    using MatrixfN3   = Matrix<float, num_keypoints, num_axis>;
    MatrixfN3 v(vertices[0].data());
    MatrixfN3 system(bboxes[0].data());
    // TODO: QR decomposition
    // devan: append '1' at the end of each row
    //auto system_h = system.rowwise().homogeneous().eval();
    // devan: QR decomposition
    //auto system_g = system_h.colPivHouseholderQr();
    //auto solution = system_g.solve(v).eval();
    //box_transformation.topLeftCorner<3, 4>() = solution.transpose();
    //Update();
    UpdateBox();
}

void SneakerDetector3D::UpdateBox() {
    // Compute the eight vertices of the bounding box from Box's parameters
    auto w = box_scale[0] / 2.f;
    auto h = box_scale[1] / 2.f;
    auto d = box_scale[2] / 2.f;
    
    // Define the local coordinate system, w.r.t. the center of the boxs
    bboxes[0] = std::array<float, 3>({0., 0., 0.});
    bboxes[1] = std::array<float, 3>({-w, -h, -d});
    bboxes[2] = std::array<float, 3>({-w, -h, +d});
    bboxes[3] = std::array<float, 3>({-w, +h, -d});
    bboxes[4] = std::array<float, 3>({-w, +h, +d});
    bboxes[5] = std::array<float, 3>({+w, -h, -d});
    bboxes[6] = std::array<float, 3>({+w, -h, +d});
    bboxes[7] = std::array<float, 3>({+w, +h, -d});
    bboxes[8] = std::array<float, 3>({+w, +h, +d});
    
    for (int i = 0; i < num_keypoints; ++i) {
        //bounding_box_[i] = transformation_.topLeftCorner<3, 3>() * bounding_box_[i] + transformation_.col(3).head<3>();
        const auto& box_trans_crop = box_transformation.TopLeftCorner<3, 3>();
        std::array<float, 3> box;
        box.fill(0);
        for(int m=0; m<3; ++m) {
            for(int n=0; n<3; ++n) {
               box[m] += box_trans_crop(m, n) * bboxes[i][n] + box_transformation(3, n);
            }
            
        }
        bboxes[i] = Vector<float, 3>(box.data());
    }
}

// devan: graphs/object_detection_3d/subgraphs/objectron_detection_gpu.pbtxt
std::shared_ptr<Mat> SneakerDetector3D::ProcessSDKInputMat(std::shared_ptr<Mat> mat, std::string name) {
    auto target_dims   = GetInputShape(name);
    auto target_height = target_dims[2];
    auto target_width  = target_dims[3];

    auto input_height  = mat->GetHeight();
    auto input_width   = mat->GetWidth();

    if (input_height != target_height || input_width !=target_width) {
        const float scale =
                  std::min(static_cast<float>(target_width) / input_width,
                           static_cast<float>(target_height) / input_height);
        const int resized_width  = std::round(input_width * scale);
        const int resized_height = std::round(input_height * scale);
        
        // TODO: we should use INTER_AREA when scale<1.0, use INTER_LINEAR for now, as TNN does not support INTER_AREA
        auto interp_mode = scale < 1.0f ? TNNInterpLinear : TNNInterpLinear;
        DimsVector intermediate_shape = {1, 4, resized_height, resized_width};
        auto intermediate_mat = std::make_shared<Mat>(DEVICE_ARM, N8UC4, intermediate_shape);
        auto status = Resize(mat, intermediate_mat, interp_mode);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);

        const int top    = (target_height - resized_height) / 2;
        const int bottom = (target_height - resized_height) - top;
        const int left   = (target_width  - resized_width) / 2;
        const int right  = (target_width  - resized_width) - left;

        auto input_mat = std::make_shared<Mat>(DEVICE_ARM, N8UC4, target_dims);
        status = CopyMakeBorder(intermediate_mat, input_mat, top, bottom, left, right, TNNBorderConstant);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);
        
        return input_mat;
    }
    return mat;
}

MatConvertParam SneakerDetector3D::GetConvertParamForInput(std::string name) {
    MatConvertParam param;
    param.scale = {2.0 / 255.0, 2.0 / 255.0, 2.0 / 255.0, 0.0};
    param.bias   = {-1.0,        -1.0,        -1.0,       0.0};
    //TODO: ensure mediapipe requires RGB or BGR
    
    return param;
}

// graphs/object_detection_3d/subgraphs/objectron_detection_gpu.pbtxt:TfLiteTensorsToObjectsCalculator::ProcessCPU
Status SneakerDetector3D::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    
    auto option = dynamic_cast<SneakerDetector3DOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    auto output = dynamic_cast<SneakerDetector3DOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));
    
    // (1, 1, 40, 30)
    auto heatmap   = output->GetMat("Identity");
    // (1, 16, 40, 30)
    auto offsetmap = output->GetMat("Identity_1");
    
    std::vector<ObjectInfo> objects;
    DecodeBoundingBoxKeypoints(heatmap, offsetmap, objects);
    
    Lift2DTo3D(objects);
    
    Project3DTo2D(objects);
    
    return status;
}

void SneakerDetector3D::DecodeBoundingBoxKeypoints(std::shared_ptr<Mat>heatmap, std::shared_ptr<Mat>offsetmap, std::vector<ObjectInfo>& objs) {
    objs.clear();

    auto heatmap_dims = heatmap->GetDims();
    auto offsetmap_dims = offsetmap->GetDims();
    constexpr int offsetmap_channel_num = 16;
    // check shapes
    if (heatmap_dims[1] != 1 || offsetmap_dims[1] != offsetmap_channel_num) {
        return;
    }
    if (heatmap_dims[2] != offsetmap_dims[2] || heatmap_dims[3] != offsetmap_dims[3]) {
        return;
    }
    
    const float offset_scale = std::min(offsetmap_dims[2], offsetmap_dims[3]);
    std::vector<std::pair<int, int>> center_points;
    ExtractCenterKeypoints(heatmap, center_points);

    constexpr int voting_radius = 2;
    const float* heatmap_data = static_cast<float *>(heatmap->GetData());
    const int heatmap_width = heatmap->GetWidth();
    
    std::vector<BeliefBox> boxes;
    const int voting_allowance = 1;
    for(const auto& center_point:center_points) {
        int center_x = center_point.first;
        int center_y = center_point.second;
        BeliefBox box;
        box.box_2d.emplace_back(center_x, center_y);
        box.belief = heatmap_data[center_y * heatmap_width + center_x];
        DecodeByVoting(heatmap, offsetmap, center_x, center_y, offset_scale, offset_scale, voting_radius, &box);
        if (IsNewBeliefBox(&boxes, &box, voting_allowance)) {
            boxes.push_back(std::move(box));
        }
    }
    
    const float x_scale = 1.0f / offsetmap->GetWidth();
    const float y_scale = 1.0f / offsetmap->GetHeight();
    for (const auto& box : boxes) {
        ObjectInfo object;
        for (const auto& point : box.box_2d) {
            object.key_points.push_back(std::make_pair(point.first * x_scale, point.second * y_scale));
        }
    }
}

void SneakerDetector3D::Lift2DTo3D(std::vector<ObjectInfo>& objects) {
    constexpr float projection_matrix[4*4] = {
        1.5731,   0,       0, 0,
        0,   2.0975,       0, 0,
        0,        0, -1.0002, -0.2,
        0,        0,      -1, 0
    };
    
    constexpr float epnp_alpha[8*4] = {
        4.0f, -1.0f, -1.0f, -1.0f,
        2.0f, -1.0f, -1.0f,  1.0f,
        2.0f, -1.0f,  1.0f, -1.0f,
        0.0f, -1.0f,  1.0f,  1.0f,
        2.0f,  1.0f, -1.0f, -1.0f,
        0.0f,  1.0f, -1.0f,  1.0f,
        0.0f,  1.0f,  1.0f, -1.0f,
        -2.0f, 1.0f,  1.0f,  1.0f
    };
    
    const float fx = projection_matrix[0 * 4 + 0];
    const float fy = projection_matrix[1 * 4 + 1];
    const float cx = projection_matrix[0 * 4 + 2];
    const float cy = projection_matrix[1 * 4 + 2];
    
    std::vector<float> m(16*12, 0);
    for (auto& object : objects) {
        //Eigen::Matrix<float, 16, 12, Eigen::RowMajor> m = Eigen::Matrix<float, 16, 12, Eigen::RowMajor>::Zero(16, 12);
        if (object.key_points.size() != 9)
            continue;
        const auto& key_points = object.key_points;
        for (int i = 0; i < 8; ++i) {
            const float kp_x = key_points[i+1].first;
            const float kp_y = key_points[i+1].second;
            // swap x and y given that our image is in portrait orientation
            float u = kp_y * 2 - 1;
            float v = kp_x * 2 - 1;
            for (int j = 0; j < 4; ++j) {
                // For each of the 4 control points, formulate two rows of the
                // m matrix (two equations).
                const float control_alpha = epnp_alpha[i * 4 + j];
                m[i * 2 * 12 + j * 3] = fx * control_alpha;
                m[i * 2 * 12 + j * 3 + 2] = (cx + u) * control_alpha;
                m[(i * 2 + 1) * 12 + j * 3 + 1] = fy * control_alpha;
                m[(i * 2 + 1) * 12 + j * 3 + 2] = (cy + v) * control_alpha;
            }
        }
        //TODO: replace eigen here, eigenvalue solver is required
        // This is a self adjoint matrix. Use SelfAdjointEigenSolver for a fast
        // and stable solution.
        Eigen::Map<Eigen::Matrix<float, 16, 12, Eigen::RowMajor>> m_(&(m[0]));
        Eigen::Matrix<float, 12, 12, Eigen::RowMajor> mt_m = m_.transpose() * m_;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 12, 12, Eigen::RowMajor>> eigen_solver(mt_m);
        if (eigen_solver.info() != Eigen::Success) {
            LOGE("eigen solver failed!\n");
            continue;
        }
        if (eigen_solver.eigenvalues().size() != 12) {
            LOGE("invalid eigenvalues!\n");
            continue;
        }
        // Eigenvalues are sorted in increasing order for SelfAdjointEigenSolver
        // only! If you use other Eigen Solvers, it's not guaranteed to be in
        // increasing order. Here, we just take the eigen vector corresponding
        // to first/smallest eigen value, since we used SelfAdjointEigenSolver.
        Eigen::VectorXf eigen_vec = eigen_solver.eigenvectors().col(0);
        Eigen::Map<Eigen::Matrix<float, 4, 3, Eigen::RowMajor>> control_matrix(
                                                                               eigen_vec.data());
        if (control_matrix(0, 2) > 0) {
            control_matrix = -control_matrix;
        }
        // First set the center keypoint.
        auto& key_points_3d = object.key_points_3d;
        key_points_3d.resize(9);
        key_points_3d[0] = std::make_tuple(control_matrix(0, 0), control_matrix(0, 1), control_matrix(0, 2));
        // Then set the 8 vertices.
        Eigen::Matrix<float, 8, 4, Eigen::RowMajor> epnp_alpha_;
        for(int h=0; h<8; ++h) {
            for(int w=0; w<4; ++w) {
                epnp_alpha_(h, w) = epnp_alpha[h * 4 + w];
            }
        }
        Eigen::Matrix<float, 8, 3, Eigen::RowMajor> vertices = epnp_alpha_ * control_matrix;
        for (int i = 0; i < 8; ++i) {
            key_points_3d[i + 1] = std::make_tuple(vertices(i, 0), vertices(i, 1), vertices(i, 2));
        }
    }
}

void SneakerDetector3D::Project3DTo2D(std::vector<ObjectInfo>& objects) {
    constexpr float projection_matrix[4*4] = {
        1.5731,   0,       0, 0,
        0,   2.0975,       0, 0,
        0,        0, -1.0002, -0.2,
        0,        0,      -1, 0
    };
    
    for (auto& object : objects) {
        int kp_idx = 0;
        auto& kp2d_list = object.key_points;
        for (auto& kp3d : object.key_points_3d) {
            float x = std::get<0>(kp3d);
            float y = std::get<1>(kp3d);
            float z = std::get<2>(kp3d);
            float p3d[4] = {x, y, z, 1};
            float projected_p3d[4];
            for(int h=0; h<4; ++h) {
                projected_p3d[h] = 0;
                for(int k=4; k<0; ++k) {
                    projected_p3d[h]+= projection_matrix[h * 4 + k] * p3d[k];
                }
            }
            const float inv_w = 1.0f / projected_p3d[3];
            float  u = (projected_p3d[1] * inv_w + 1.0f) * 0.5f;
            float  v = (projected_p3d[0] * inv_w + 1.0f) * 0.5f;
            kp2d_list[kp_idx++] = std::make_pair(u, v);
        }
    }
}

}
