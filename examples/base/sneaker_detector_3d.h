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

#ifndef TNN_EXAMPLES_BASE_SNEAKER_DETECTOR_3D_H_
#define TNN_EXAMPLES_BASE_SNEAKER_DETECTOR_3D_H_

#include "tnn_sdk_sample.h"
#include "tnn/utils/dims_vector_utils.h"
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <utility>

namespace TNN_NS {

class SneakerDetector3DInput : public TNNSDKInput {
public:
    SneakerDetector3DInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~SneakerDetector3DInput(){}
};

class SneakerDetector3DOutput : public TNNSDKOutput {
public:
    SneakerDetector3DOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~SneakerDetector3DOutput();
    std::vector<ObjectInfo> object_list;
};


class SneakerDetector3DOption : public TNNSDKOption {
public:
    SneakerDetector3DOption() {}
    virtual ~SneakerDetector3DOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
    
    
};

template <typename T, int N>
class Vector {
private:
    using ThisVector = Vector<T, N>;
    using InnerArray = std::array<T, N>;
public:
    Vector() {}
    Vector(InnerArray array ): data_(std::move(array)) {}
    
    Vector(const T* p) {
        memcpy(data_.data(), p, sizeof(T)*N);
    }

    ThisVector operator - (const ThisVector& vec) const {
        ThisVector rst;
        for(int i=0; i<N; ++i) {
            rst[i] = (*this)[i] - vec[i];
        }
        return rst;
    }

    T norm() const {
        T squre_sum = 0;
        for(int i=0; i<N; ++i) {
            squre_sum += data_[i] * data_[i];
        }
        return sqrt(squre_sum);
    }
    
    const T& operator[] (int i) const {
        return data_[i];
    }
    
    T& operator[] (int i) {
        return data_[i];
    }

    void SetZero() {
        data_.fill(0);
    }
    
    const T* data() const {
        return data_.data();
    }

private:
    std::array<T, N> data_;
};

template <typename T, int M, int N>
class Matrix {
private:
    using ThisMat = Matrix<T, M, N>;
    using InnerArray = std::array<T, M*N>;
public:
    Matrix() {}
    Matrix(InnerArray array): data_(std::move(array)) {}
    Matrix(const T* p) {
        memcpy(data_.data(), p, sizeof(T)*M*N);
    }

    ~Matrix() {}

    void SetIdentity() {
        // only valid when M==N
        if(M != N)
            return;
        SetZero();
        for(int i=0; i<M; ++i)
            data_[i * N + i] = static_cast<T>(1);
    }

    void SetZero() {
        data_.fill(0);
    }

    void SetTopLeftRegion(const T* new_data, int rm, int rn, int stride_n) {
        if (rm > M || rn > N)
            return;

        for(int m=0; m<rm; ++m) {
            for(int n=0; n<rn; ++n) {
                data_[m * N + n] = new_data[m * stride_n + n];
            }
        }
    }
    
    template <int RM, int RN>
    Matrix<T, RM, RN> TopLeftCorner() {
        if (RM > M || RN > N)
            return {};
        
        Matrix<T, RM, RN> rst;
        rst.SetTopLeftRegion(this->data_.data(), RM, RN, N);
        
        return rst;
    }
    
    const T* data() const {
        return data_.data();
    }
    
    const T& operator() (int m, int n) const {
        return  data_[m * N + n];
    }
    
    T& operator() (int m, int n) {
        return data_[m * N + n];
    }
    
private:
    std::array<T, M*N> data_;
};

class SneakerDetector3D : public TNN_NS::TNNSDKSample {
private:
    using Vectorf3 = Vector<float, 3>;
    using Matrixf4 = Matrix<float, 4, 4>;
public:
    ~SneakerDetector3D();
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                            std::string name = kTNNSDKDefaultName);
private:
    // detector post-processing
    void DecodeBoundingBoxKeypoints(std::shared_ptr<Mat>heatmap, std::shared_ptr<Mat>offsetmap, std::vector<ObjectInfo>& objs);
    void Lift2DTo3D(std::vector<ObjectInfo>& objects);
    void Project3DTo2D(std::vector<ObjectInfo>& objects);
    // for box
    void FitBox(const std::vector<Vectorf3>& vertices);
    void UpdateBox();
    Vectorf3 box_scale;
    Matrixf4 box_transformation;
    std::vector<std::array<int, 2>>box_edges = {
        // Add the edges in the cube, they are sorted according to axis (x-y-z).
        {1, 5},
        {2, 6},
        {3, 7},
        {4, 8},

        {1, 3},
        {5, 7},
        {2, 4},
        {6, 8},

        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8}
    };
    std::vector<Vector<float, 3>> bboxes;
    static constexpr int num_axis = 3;
    static constexpr int num_edges_per_axis = 4;
    // the number of 3d keypoints
    static constexpr int num_keypoints = 9;
};

}
#endif // TNN_EXAMPLES_BASE_SNEAKER_DETECTOR_3D_H_

