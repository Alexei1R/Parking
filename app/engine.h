//
// Created by toor on 2/16/24.
//

#ifndef ENGINE_H
#define ENGINE_H

#ifdef NDEBUG

#include "pch.h"


#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <chrono>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include "Utils.h"

using preciseStopwatch = Stopwatch<>;

// Precision used for GPU inference
enum class Precision {
    FP32,
};

// Options for the network
struct Options {
    Precision precision = Precision::FP32;
    int32_t calibrationBatchSize = 128;
    int32_t optBatchSize = 1;
    int32_t maxBatchSize = 16;
    int deviceIndex = 0;
};

class Engine {
public:
    Engine(const Options &options);

    ~Engine();

    bool buildLoadNetwork(std::string onnxModelPath);

    bool loadNetwork(std::string trtModelPath);

//    void RunInference
//
//    Model supports dynamic batch size
//            Input name: image_input
//            Input dims: -1 224 224 3
//    Input name: tabular_input
//            Input dims: -1 182
//how i create the function in my code
    bool RunInference(cv::cuda::GpuMat& image, std::vector<float>& lidarData);



private:
    bool build(std::string onnxModelPath);

    std::string serializeEngineOptions(const Options &options, const std::string &onnxModelPath);

    void clearGpuBuffers();

    void getDeviceNames(std::vector<std::string> &deviceNames);


    // Holds pointers to the input and output GPU buffers
    std::vector<void *> m_buffers;
    std::vector<uint32_t> m_outputLengths{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;
    int32_t m_inputBatchSize;


    int m_LidarDataSize ;

    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    const Options m_options;
    Logger m_logger;
};

#endif
#endif //ENGINE_H
