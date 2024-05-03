//
// Created by toor on 2/16/24.
//

#include "engine.h"

//tensorflow model
//# Image processing branch
//image_input = Input(shape=(224, 224, 3))
//base_model = ResNet50(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
//x = base_model(image_input)
//x = Flatten()(x)
//x = Dense(128, activation='relu')(x)
//image_branch_output = Dense(64, activation='relu')(x)
//
//# Tabular data processing branch
//tabular_input = Input(shape=(182,))
//tabular_input_expanded = tf.expand_dims(tabular_input, axis=1)  # Add the sequence dimension for LSTM
//        y = LSTM(128, return_sequences=True)(tabular_input_expanded)
//y = LSTM(64, return_sequences=False)(y)
//tabular_branch_output = Dense(64, activation='relu')(y)
//
//# Combine the outputs of the two branches
//combined = concatenate([image_branch_output, tabular_branch_output])
//z = Dense(64, activation='relu')(combined)
//z = Dense(32, activation='relu')(z)
//
//# Output layer with two units: CarSpeed and CarSteering
//output = Dense(2, activation='linear')(z)
//
//# Build the model
//model = Model(inputs=[image_input, tabular_input], outputs=output)
//model.compile(optimizer='adam', loss='mse', metrics=['mae'])
//model.summary()


#ifdef NDEBUG

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>

using namespace nvinfer1;
using namespace Util;

std::vector<std::string> Util::getFilesInDirectory(const std::string &dirPath) {
    std::vector<std::string> filepaths;
    for (const auto &entry: std::filesystem::directory_iterator(dirPath)) {
        filepaths.emplace_back(entry.path().string());
    }
    return filepaths;
}


Engine::Engine(const Options &options) : m_options(options) {}

Engine::~Engine() {
    clearGpuBuffers();
}

void Engine::clearGpuBuffers() {
    if (!m_buffers.empty()) {
        // Free GPU memory of outputs
        const auto numInputs = m_inputDims.size();
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
            Util::checkCudaErrorCode(cudaFree(m_buffers[outputBinding]));
        }
        m_buffers.clear();
    }
}

bool Engine::buildLoadNetwork(std::string onnxModelPath) {
    const auto engineName = serializeEngineOptions(m_options, onnxModelPath);
    std::cout << "Searching for engine file with name: " << engineName << std::endl;

    if (Util::doesFileExist(engineName)) {
        std::cout << "Engine found, not regenerating..." << std::endl;
    } else {
        if (!Util::doesFileExist(onnxModelPath)) {
            throw std::runtime_error("Could not find onnx model at path: " + onnxModelPath);
        }

        // Was not able to find the engine file, generate...
        std::cout << "Engine not found, generating. This could take a while..." << std::endl;

        // Build the onnx model into a TensorRT engine
        auto ret = build(onnxModelPath);
        if (!ret) {
            return false;
        }
    }
    std::cout << " Engine file found, loading..." << std::endl;
    // Load the TensorRT engine file into memory
    return loadNetwork(engineName);
}

bool
Engine::build(std::string onnxModelPath) {
    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const auto numInputs = network->getNbInputs();
    if (numInputs < 1) {
        throw std::runtime_error("Error, model needs at least 1 input!");
    }
    const auto input0Batch = network->getInput(0)->getDimensions().d[0];
    for (int32_t i = 1; i < numInputs; ++i) {
        if (network->getInput(i)->getDimensions().d[0] != input0Batch) {
            throw std::runtime_error("Error, the model has multiple inputs, each "
                                     "with differing batch sizes!");
        }
    }

    // Check to see if the model supports dynamic batch size or not
    bool doesSupportDynamicBatch = false;
    if (input0Batch == -1) {
        doesSupportDynamicBatch = true;
        std::cout << "Model supports dynamic batch size" << std::endl;
    } else {
        std::cout << "Model only supports fixed batch size of " << input0Batch << std::endl;
        // If the model supports a fixed batch size, ensure that the maxBatchSize
        // and optBatchSize were set correctly.
        if (m_options.optBatchSize != input0Batch || m_options.maxBatchSize != input0Batch) {
            throw std::runtime_error("Error, model only supports a fixed batch size of " + std::to_string(input0Batch) +
                                     ". Must set Options.optBatchSize and Options.maxBatchSize to 1");
        }
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    auto profile = builder->createOptimizationProfile();
    for (int i = 0; i < numInputs; ++i) {
        auto input = network->getInput(i);
        auto dims = input->getDimensions();
        if (dims.nbDims == 4) {  // For image inputs
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
                                   nvinfer1::Dims4(1, dims.d[1], dims.d[2], dims.d[3]));
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
                                   nvinfer1::Dims4(1, dims.d[1], dims.d[2], dims.d[3]));
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
                                   nvinfer1::Dims4(1, dims.d[1], dims.d[2], dims.d[3]));
        } else if (dims.nbDims == 2) {  // For vector inputs
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, dims.d[1]));
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, dims.d[1]));
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1, dims.d[1]));
        }



        //Print input name and dims
        std::cout << "Input name: " << input->getName() << std::endl;
        std::cout << "Input dims: ";
        for (int d = 0; d < dims.nbDims; ++d) {
            std::cout << dims.d[d] << " ";
        }
        std::cout << std::endl;

    }
    config->addOptimizationProfile(profile);


    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);


    //Build the engine
    std::cout << "Building the engine..." << std::endl;
    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }



    // Set the precision level
    const auto engineName = serializeEngineOptions(m_options, onnxModelPath);

    // Write the engine to disk
    std::ofstream outfile(engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << engineName << std::endl;

    Util::checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return true;
}


std::string Engine::serializeEngineOptions(const Options &options, const std::string &onnxModelPath) {
    const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
    std::string engineName =
            onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos) + ".engine";

    // Add the GPU device name to the file to ensure that the model is only used
    // on devices with the exact same GPU
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);

    if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
        throw std::runtime_error("Error, provided device index is out of range!");
    }

    auto deviceName = deviceNames[options.deviceIndex];
    // Remove spaces from the device name
    deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());

    engineName += "." + deviceName;

    engineName += ".fp32";

    engineName += "." + std::to_string(options.maxBatchSize);
    engineName += "." + std::to_string(options.optBatchSize);

    return engineName;
}

void Engine::getDeviceNames(std::vector<std::string> &deviceNames) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device = 0; device < numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        deviceNames.push_back(std::string(prop.name));
    }
}

bool Engine::loadNetwork(std::string trtModelPath) {
    // Read the serialized model from disk
    if (!Util::doesFileExist(trtModelPath)) {
        std::cout << "Error, unable to read TensorRT model at path: " + trtModelPath << std::endl;
        return false;
    } else {
        std::cout << "Loading TensorRT engine file at path: " << trtModelPath << std::endl;
    }

    std::ifstream file(trtModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(m_logger)};
    if (!m_runtime) {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
                      ". Note, your device has " +
                      std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    // The execution context contains all of the state associated with a
    // particular invocation
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    clearGpuBuffers();
    m_buffers.resize(m_engine->getNbIOTensors());

    m_outputLengths.clear();
    m_inputDims.clear();
    m_outputDims.clear();
    m_IOTensorNames.clear();

    // Create a cuda stream
    cudaStream_t stream;
    Util::checkCudaErrorCode(cudaStreamCreate(&stream));

    // Allocate GPU memory for input and output buffers
    m_outputLengths.clear();
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const auto tensorName = m_engine->getIOTensorName(i);
        std::cout << "Allocating memory for tensor: " << tensorName << std::endl;
        m_IOTensorNames.emplace_back(tensorName);
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getTensorShape(tensorName);
        const auto tensorDataType = m_engine->getTensorDataType(tensorName);

        if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
            // The implementation currently only supports inputs of type float
            if (m_engine->getTensorDataType(tensorName) != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error("Error, the implementation currently only supports float inputs");
            }


            if (std::string(tensorName) == "tabular_input") {
                m_LidarDataSize = tensorShape.d[1];
                //print input array size
                std::cout << "Input Lidar size: " << m_LidarDataSize << std::endl;
                // Allocate memory for the input buffer
                Util::checkCudaErrorCode(
                        cudaMallocAsync(&m_buffers[i], m_options.maxBatchSize * m_LidarDataSize * sizeof(float),
                                        stream));

            }

            if(std::string(tensorName) == "image_input"){
                // Allocate memory for the input buffer
                Util::checkCudaErrorCode(
                        cudaMallocAsync(&m_buffers[i], m_options.maxBatchSize * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3] * sizeof(float),
                                        stream));
            }

            // Store the input dims for later use
            m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
            m_inputBatchSize = tensorShape.d[0];
        } else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
            //PRINT OUTPUT TENSOR NAME
            std::cout << "Output tensor name: " << tensorName << std::endl;
            //PRINT OUTPUT TENSOR DIMS
            std::cout << "Output tensor dims: ";
            for (int d = 0; d < tensorShape.nbDims; ++d) {
                std::cout << tensorShape.d[d] << " ";
            }
            std::cout << std::endl;
            //PRINT DATA TYPE
            if (tensorDataType == nvinfer1::DataType::kFLOAT) {
                std::cout << "Output tensor data type: FLOAT" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kHALF) {
                std::cout << "Output tensor data type: HALF" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kINT8) {
                std::cout << "Output tensor data type: INT8" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kINT32) {
                std::cout << "Output tensor data type: INT32" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kBOOL) {
                std::cout << "Output tensor data type: BOOL" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kUINT8) {
                std::cout << "Output tensor data type: UINT8" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kFP8) {
                std::cout << "Output tensor data type: FP8" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kINT8) {
                std::cout << "Output tensor data type: INT8" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kINT8) {
                std::cout << "Output tensor data type: INT8" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kINT8) {
                std::cout << "Output tensor data type: INT8" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kINT8) {
                std::cout << "Output tensor data type: INT8" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kINT8) {
                std::cout << "Output tensor data type: INT8" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kINT8) {
                std::cout << "Output tensor data type: INT8" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kINT8) {
                std::cout << "Output tensor data type: INT8" << std::endl;
            } else if (tensorDataType == nvinfer1::DataType::kINT8) {
                std::cout << "Output tensor data type: INT8" << std::endl;
            }


            std::cout << "Binding is an output" << std::endl;
            // The binding is an output
            uint32_t outputLength = 1;
            m_outputDims.push_back(tensorShape);

            for (int j = 1; j < tensorShape.nbDims; ++j) {
                // We ignore j = 0 because that is the batch size, and we will take that
                // into account when sizing the buffer
                outputLength *= tensorShape.d[j];
            }

            m_outputLengths.push_back(outputLength);
            // Now size the output buffer appropriately, taking into account the max
            // possible batch size (although we could actually end up using less
            // memory)
//            Util::checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLength * m_options.maxBatchSize * sizeof(T), stream));
            // Allocate memory for the output buffer
            if (tensorDataType == nvinfer1::DataType::kFLOAT) {

                Util::checkCudaErrorCode(
                        cudaMallocAsync(&m_buffers[i], outputLength * m_options.maxBatchSize * sizeof(float), stream));
                std::cout << "Allocated memory for output buffer" << std::endl;

            } else {
                std::cout << "Model only supports float output data type" << std::endl;
                //TODO: Add support for other data types
            }
        } else {
            throw std::runtime_error("Error, IO Tensor is neither an input or output!");
        }

    }
    return true;
}

bool Engine::RunInference(cv::cuda::GpuMat &image, std::vector<float> &lidarData) {
    // Check if engine is built and loaded
    if (!m_engine || !m_context) {
        std::cerr << "Engine or context is not initialized properly" << std::endl;
        return false;
    }
    const auto numInputs = m_inputDims.size();
    std::cout << "Number of inputs: " << numInputs << std::endl;

//    Model supports dynamic batch size
//            Input name: image_input
//            Input dims: -1 224 224 3
//    Input name: tabular_input
//            Input dims: -1 182

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // Preprocess all the inputs
    for (size_t i = 0; i < numInputs; ++i) {
        const auto &dims = m_inputDims[i];
        const auto name = m_IOTensorNames[i];


        if (std::string(name) == "image_input") {
            std::cout << "Input name: " << name << std::endl;

            int rows = dims.d[0];
            int cols = dims.d[1];
            int channels = dims.d[2];

            if (image.rows != rows || image.cols != cols || image.channels() != channels) {
                std::cerr << "Error, image dimensions do not match the expected input dimensions" << std::endl;
                //Get and Expected input dimensions
                std::cerr << "Expected input dimensions: " << rows << "x" << cols << "x" << channels << std::endl;
                std::cerr << "Actual input dimensions: " << image.rows << "x" << image.cols << "x" << image.channels()
                          << std::endl;
                return false;
            }

            nvinfer1::Dims4 inputDims = nvinfer1::Dims4(1, rows, cols, channels);
            m_context->setInputShape(name.c_str(), inputDims);

            //convert image from hwc to chw
            cv::cuda::GpuMat gpu_dst(1, image.rows * image.cols * image.channels(), CV_8UC3);

            size_t width = image.cols * image.rows;
            std::vector<cv::cuda::GpuMat> input_channels{
                    cv::cuda::GpuMat(image.rows, image.cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * 0])),
                    cv::cuda::GpuMat(image.rows, image.cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * 0])),
                    cv::cuda::GpuMat(image.rows, image.cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * 0]))};
            cv::cuda::split(image, input_channels); // HWC -> CHW

            cv::cuda::GpuMat mfloat;
            gpu_dst.convertTo(mfloat, CV_32FC3);



            //Copy the image to the GPU buffer
            Util::checkCudaErrorCode(cudaMemcpyAsync(m_buffers[i], mfloat.ptr<float>(), rows * cols * channels * sizeof(float),
                                                     cudaMemcpyHostToDevice, inferenceCudaStream));


        }
        if (std::string(name) == "tabular_input") {
            std::cout << "Input name: " << name << std::endl;

            int arraySize = dims.d[0];
            std::cout << "Input Lidar size: " << arraySize << std::endl;
            if (lidarData.size() != arraySize) {
                std::cerr << "Error, lidar data size does not match the expected input size" << std::endl;
                std::cerr << "Expected input size: " << arraySize << std::endl;
                std::cerr << "Actual input size: " << lidarData.size() << std::endl;
                return false;
            }

            nvinfer1::Dims2 inputDims = nvinfer1::Dims2(1, arraySize);
            m_context->setInputShape(name.c_str(), inputDims);

            std::cout << "Copying lidar data to GPU" << std::endl;
            // Copy the lidar data to the GPU


//            Copies data between host and device
//            Copies count bytes from the memory area pointed to by src to the memory area pointed to by dst, where kind specifies the direction of the copy, and must be one of ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost, ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing ::cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer values. However, ::cudaMemcpyDefault is only allowed on systems that support unified virtual addressing. The memory areas may not overlap. Calling ::cudaMemcpyAsync() with dst and src pointers that do not match the direction of the copy results in an undefined behavior. ::cudaMemcpyAsync() is asynchronous with respect to the host, so the call may return before the copy is complete. The copy can optionally be associated to a stream by passing a non-zero stream argument. If kind is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and the stream is non-zero, the copy may overlap with operations in other streams. The device version of this function only handles device to device copies and cannot be given local or shared pointers.
            Util::checkCudaErrorCode(cudaMemcpyAsync(m_buffers[i], &lidarData[0], arraySize * sizeof(float),
                                                     cudaMemcpyHostToDevice, inferenceCudaStream));
            std::cout << "Coooooooooooooooooooopied lidar data to GPU" << std::endl;

        }


    }


// Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    } else {
        std::cout << "All required dimensions specified" <<
                  std::endl;
    }

    //print buffer info
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        std::cout << "Buffer " << i << " address: " << m_buffers[i] << std::endl;
        std::cout << "Tensor name: " << m_IOTensorNames[i] << std::endl;
    }

// Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return false;
        }
    }

// Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    } else{
        std::cout << "Inference completed" << std::endl;
    }


    //print model output
//    Output tensor name: dense_5
//    Output tensor dims: 1 2
//    Output tensor data type: FLOAT

    std::vector<float> outputData(m_outputLengths[0]);

    cudaMemcpyAsync(outputData.data(), m_buffers[numInputs], m_outputLengths[0] * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream);
    //print output data
    std::cout << "Output data: ";
    for (int i = 0; i < m_outputLengths[0]; ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;


    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));


    return true;

}


#endif
