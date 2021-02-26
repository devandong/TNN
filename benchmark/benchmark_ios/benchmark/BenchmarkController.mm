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

#import "BenchmarkController.h"
#import <tnn/tnn.h>
#include <fstream>
#include <cmath>
#include <sys/time.h>
#include <float.h>
#include <sstream>
#include <map>

using namespace std;
using namespace TNN_NS;

struct BenchModel {
    string name;
    string tnn_proto_content;
    string tnn_model_content;
    string coreml;
};

struct BenchOption {
    int warm_count = 10;
    int forward_count = 20;
    int create_count = 1;
    
    string description() {
        ostringstream ostr;
        ostr << "create_count = " << create_count
        << "  warm_count = " << warm_count
        << "  forward_count = " << forward_count;
        
        ostr << std::endl;
        return ostr.str();
    };
};

struct BenchResult {
    Status status;
    
    //time
    float min = FLT_MAX;
    float max = FLT_MIN;
    float avg = 0;
    float total = 0;
    int count = 0;
    
    float diff = 0;
    
    int addTime(float time){
        count++;
        total += time;
        min = std::min(min, time);
        max = std::max(max, time);
        avg = total/count;
        return 0;
    };
    
    string description() {
        ostringstream ostr;
        ostr << "min = " << min << "  max = " << max << "  avg = " <<avg;
        
        if (status != TNN_OK) {
            ostr << "\nerror = "<<status.description();
        }
        ostr << std::endl;
        
        return ostr.str();
    };
};
// save files on device
void SaveMatToFile(Mat &mat, const std::string &filename) {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    NSString *filePath = [documentsDirectory stringByAppendingPathComponent:[NSString stringWithCString:filename.c_str()
                                                                                               encoding:[NSString defaultCStringEncoding]]];
    std::stringstream ss;
    const auto &dims = mat.GetDims();
    // shape
    for(int idx = 0; idx < dims.size(); ++idx) {
        ss << dims[idx];
        if (idx != dims.size()-1)
            ss << ",";
        else
            ss << endl;
    }
    // data
    const float *data = reinterpret_cast<float *>(mat.GetData());
    for(int idx=0; idx < DimsVectorUtils::Count(dims); ++idx) {
        ss << data[idx] << endl;
    }
    
    NSString *content = [NSString stringWithCString:ss.str().c_str() encoding:[NSString defaultCStringEncoding]];
    [content writeToFile:filePath atomically:NO
                encoding:[NSString defaultCStringEncoding]
                   error:nil];
}

using MatMap = std::map<std::string, std::shared_ptr<Mat>>;
// create Mat according to blob
MatMap CreateMatMap(BlobMap &blob_map, DeviceType device = DEVICE_ARM) {
    MatMap mat_map;
    for(auto &iter : blob_map) {
        const auto &name = iter.first;
        Blob *device_blob = iter.second;
        BlobDesc &blob_desc = device_blob->GetBlobDesc();
        
        //DataType data_type = DATA_TYPE_FLOAT;
        MatType mat_type   = NCHW_FLOAT;
        
        //size_t bytes = DimsVectorUtils::Count(blob_desc.dims) * DataTypeUtils::GetBytesSize(data_type);
        auto mat = std::make_shared<Mat>(device, mat_type, blob_desc.dims);
        mat_map[name] = mat;
    }
    return mat_map;
}

// Init MatMap
bool InitMatMap(MatMap &mat_map, bool save_mat, const std::vector<std::string> &filepath_list) {
    if (filepath_list.size()!=0 && mat_map.size() != filepath_list.size())
        return false;
    
    bool use_file = filepath_list.size() != 0;
    
    int idx = 0;
    for(auto iter=mat_map.begin(); iter!=mat_map.end(); ++iter, ++idx) {
        const auto &name = iter->first;
        auto mat = iter->second;
        if (mat->GetMatType() != NCHW_FLOAT)
            return false;
        
        auto data_count = DimsVectorUtils::Count(mat->GetDims());
        void *mat_data  = mat->GetData();
        if (use_file) {
            std::ifstream input_stream(filepath_list[idx]);
            if (!input_stream)
                return false;
            for(auto i=0; i<data_count; ++i) {
                input_stream >> reinterpret_cast<float *>(mat_data)[i];
            }
        } else {
            srand(time(nullptr));
            for(auto i=0; i<data_count; ++i) {
                reinterpret_cast<float *>(mat_data)[i] = static_cast<float>((rand() % 256 - 128)) / 128.0;
            }
        }
    }
    
    if (save_mat && !use_file) {
        for(auto iter=mat_map.begin(); iter != mat_map.end(); ++iter) {
            const auto &name = iter->first;
            auto mat         = iter->second;
            SaveMatToFile(*mat, name+".txt");
        }
    }
    
    return true;
}

using BlobConverterMap = std::map<std::string, std::shared_ptr<BlobConverter>>;
BlobConverterMap CreateBlobConverterMap(BlobMap &blob_map) {
    std::map<std::string, std::shared_ptr<BlobConverter>> converter_map;
    for(auto &iter : blob_map) {
        auto blob_converter = std::make_shared<BlobConverter>(iter.second);
        converter_map[iter.first] = blob_converter;
    }
    
    return converter_map;
}

bool SetInputBlobs(Instance *instance, bool save_input=false, const std::vector<std::string> &filepath_list={}) {
    BlobMap input_blob_map;
    auto status = instance->GetAllInputBlobs(input_blob_map);
    RETURN_VALUE_ON_NEQ(status, TNN_OK, false);
    
    MatMap input_mat_map = CreateMatMap(input_blob_map);
    if (!InitMatMap(input_mat_map, save_input, filepath_list))
        return false;
    
    BlobConverterMap blob_converter_map = CreateBlobConverterMap(input_blob_map);
    
    void *command_queue;
    status = instance->GetCommandQueue(&command_queue);
    RETURN_VALUE_ON_NEQ(status, TNN_OK, false);
    
    for(auto &converter : blob_converter_map) {
        const auto &name = converter.first;
        auto blob_converter = converter.second;
        status = blob_converter->ConvertFromMat(*input_mat_map[name], MatConvertParam(), command_queue);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, false);
    }
    
    return true;
}

MatMap GetOutputBlobs(Instance *instance, const std::vector<std::string> &filepath_list={}) {
    BlobMap output_blob_map;
    auto status = instance->GetAllOutputBlobs(output_blob_map);
    RETURN_VALUE_ON_NEQ(status, TNN_OK, MatMap());
    
    MatMap output_map_map = CreateMatMap(output_blob_map);
    
    BlobConverterMap blob_converter_map = CreateBlobConverterMap(output_blob_map);
    
    void *command_queue;
    status = instance->GetCommandQueue(&command_queue);
    RETURN_VALUE_ON_NEQ(status, TNN_OK, MatMap());
    
    for(auto &converter : blob_converter_map) {
        const auto &name = converter.first;
        auto blob_converter = converter.second;
        status = blob_converter->ConvertToMat(*output_map_map[name], MatConvertParam(), command_queue);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, MatMap());
    }
    
    return output_map_map;
}

bool SaveOutputMatMap(MatMap &mat_map) {
    for(auto &iter : mat_map) {
        const auto &name = iter.first;
        auto &mat = iter.second;
        SaveMatToFile(*mat, name+".txt");
    }
    
    return true;
}

@interface BenchmarkController () {
}
@property (nonatomic, weak) IBOutlet UIButton *btnBenchmark;
@property (nonatomic, weak) IBOutlet UITextView *textViewResult;
@end

@implementation BenchmarkController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    
    
}

- (vector<BenchModel>)getAllModels {
    NSString *modelZone = [[NSBundle mainBundle] pathForResource:@"align"
                                                          ofType:nil];
    NSArray *modelList = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:modelZone
                                                                             error:nil];
    
    NSPredicate *predicateProto = [NSPredicate predicateWithFormat:@"self ENDSWITH 'proto'"];
    NSPredicate *predicateModel = [NSPredicate predicateWithFormat:@"self ENDSWITH 'model'"];
    NSPredicate *predicateCoreML = [NSPredicate predicateWithFormat:@"self ENDSWITH 'mlmodelc'"];
    
    vector<BenchModel> netmodels;
    
    for (NSString *modelDir in modelList) {
//        if (![modelDir hasPrefix:@"mobilenetv1-ssd"]) {
//            continue;
//        }
       NSString *modelDirPath = [modelZone stringByAppendingPathComponent:modelDir];
       BOOL isDirectory = NO;

       if ([[NSFileManager defaultManager] fileExistsAtPath:modelDirPath
                                                isDirectory:&isDirectory]) {
           if (!isDirectory) {
               continue;
           }
           
           BenchModel model;
           model.name = modelDir.UTF8String;
           
           NSArray *modelFiles = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:modelDirPath
                                                                                     error:nil];
           NSArray<NSString *> *protos = [modelFiles filteredArrayUsingPredicate:predicateProto];
           if (protos.count > 0) {
               auto proto = [NSString stringWithContentsOfFile:[modelDirPath stringByAppendingPathComponent:protos[0]]
                                                        encoding:NSUTF8StringEncoding
                                                           error:nil];
               if (proto.length > 0) {
                   model.tnn_proto_content = proto.UTF8String;
               }
           }
           NSArray<NSString *> *models = [modelFiles filteredArrayUsingPredicate:predicateModel];
           if (models.count > 0) {
//               model.tnn_model_content = [modelDirPath stringByAppendingPathComponent:models[0]].UTF8String;
               NSData *data = [NSData dataWithContentsOfFile:[modelDirPath
                                                              stringByAppendingPathComponent:models[0]]];
               model.tnn_model_content = string((const char *)[data bytes], [data length]);
           }
           NSArray<NSString *> *coremls = [modelFiles filteredArrayUsingPredicate:predicateCoreML];
           if (coremls.count > 0) {
               model.coreml = [modelDirPath stringByAppendingPathComponent:coremls[0]].UTF8String;
           }
           netmodels.push_back(model);
       }
    }
    return netmodels;
}

- (IBAction)onBtnBenchmark:(id)sender {
    //check release mode at Product->Scheme when running
    //运行时请在Product->Scheme中确认意见调整到release模式
    
    //搜索model目录下的所有模型
    auto allModels = [self getAllModels];
    
    BenchOption option;
    option.warm_count = 0;
    option.forward_count = 1;
    option.create_count = 1;
    
    //Get metallib path from app bundle
    //PS：A script(Build Phases -> Run Script) is added to copy the metallib file in tnn framework project to benchmark app
    //注意：此工程添加了脚本将tnn工程生成的tnn.metallib自动复制到app内
    auto pathLibrary = [[NSBundle mainBundle] pathForResource:@"tnn.metallib"
                                                       ofType:nil];
    pathLibrary = pathLibrary ? pathLibrary : @"";
    
    NSString *allResult = [NSString string];
    for (auto model : allModels) {
        NSLog(@"model: %s", model.name.c_str());
        allResult = [allResult stringByAppendingFormat:@"model: %s\n", model.name.c_str()];
        /*
        //benchmark on arm cpu
        auto result_arm = [self benchmarkWithProtoContent:model.tnn_proto_content
                                                model:model.tnn_model_content
                                               coreml:model.coreml
                                              library:pathLibrary.UTF8String
                                              netType:NETWORK_TYPE_DEFAULT
                                              deviceType:DEVICE_ARM
                                               option:option];
        NSLog(@"arm: \ntime: %s", result_arm.description().c_str());
        allResult = [allResult stringByAppendingFormat:@"arm: \ntime: %s",
                     result_arm.description().c_str()];
        */
        
        //benchmark on gpu
        auto result_gpu = [self benchmarkWithProtoContent:model.tnn_proto_content
                                                model:model.tnn_model_content
                                               coreml:model.coreml
                                              library:pathLibrary.UTF8String
                                              netType:NETWORK_TYPE_DEFAULT
                                              deviceType:DEVICE_METAL
                                               option:option];
        NSLog(@"gpu: \ntime: %s", result_gpu.description().c_str());
        allResult = [allResult stringByAppendingFormat:@"gpu: \ntime: %s\n",
                     result_gpu.description().c_str()];
    }
    
    self.textViewResult.text = allResult;
}

- (BenchResult)benchmarkWithProtoContent:(string)protoContent
                                   model:(string)modelPathOrContent
                                  coreml:(string)coremlDir
                                 library:(string)metallibPath
                                 netType:(NetworkType)net_type
                              deviceType:(DeviceType)device_type
                                  option:(BenchOption)option {
    BenchResult result;
    
    net_type = net_type == NETWORK_TYPE_COREML ? NETWORK_TYPE_COREML : NETWORK_TYPE_DEFAULT;
    
    //network init
    //网络初始化
    TNN net;
    {
        ModelConfig config;
        if (net_type == NETWORK_TYPE_COREML) {
            config.model_type = MODEL_TYPE_COREML;
            config.params = {coremlDir};
        } else {
            config.model_type = MODEL_TYPE_TNN;
            config.params = {protoContent, modelPathOrContent};
        }
        
        if (net_type == NETWORK_TYPE_COREML) {
            config.model_type = MODEL_TYPE_COREML;
        }
        
        result.status = net.Init(config);
        if (result.status != TNN_OK) {
            NSLog(@"net.Init Error: %s", result.status.description().c_str());
            return result;
        }
    }
    
    //create instance
    //创建实例instance
    std::shared_ptr<TNN_NS::Instance> instance = nullptr;
    {
        NetworkConfig network_config;
        network_config.network_type = net_type;
        network_config.library_path = {metallibPath};
        network_config.device_type =  device_type;
        instance = net.CreateInst(network_config, result.status);
        if (result.status != TNN_OK || !instance) {
            NSLog(@"net.CreateInst Error: %s", result.status.description().c_str());
            return result;
        }
    }
    
    // set input
    if(!SetInputBlobs(instance.get(), true)){
        result.status = Status(TNNERR_PARAM_ERR, "Set input blobs error!");
        return result;
    }
    
    //warm cpu, only used when benchmark
    for (int cc=0; cc<option.warm_count; cc++) {
        result.status = instance->Forward();
        if (result.status != TNN_OK) {
            NSLog(@"instance.Forward Error: %s", result.status.description().c_str());
            return result;
        }
    }
    
    //inference
    //前向推断
    bool profile_layer_time = false;
#if TNN_PROFILE
    if (profile_layer_time) {
        instance->StartProfile();
    }
#endif
    for (int cc=0; cc<option.forward_count; cc++) {
        timeval tv_begin, tv_end;
        gettimeofday(&tv_begin, NULL);
        
        result.status = instance->Forward();
        
        gettimeofday(&tv_end, NULL);
        double elapsed = (tv_end.tv_sec - tv_begin.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_begin.tv_usec) / 1000.0;
        result.addTime(elapsed);
    }
#if TNN_PROFILE
    if (profile_layer_time) {
        instance->FinishProfile(true);
    }
#endif
    
    // get output
    MatMap output_mat_map = GetOutputBlobs(instance.get());
    if (output_mat_map.size() <= 0) {
        result.status = Status(TNNERR_PARAM_ERR, "Get output blobs error!");
        return result;
    }
    
    if (!SaveOutputMatMap(output_mat_map)) {
        result.status = Status(TNNERR_PARAM_ERR, "Save output blobs error!");
        return result;
    }

    return result;
}

@end

