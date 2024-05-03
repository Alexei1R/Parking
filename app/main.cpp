#include <iostream>
#include "engine.h"
#include "filesystem"


#ifdef NDEBUG
#define BUILD_TYPE "Release"
#else
#define BUILD_TYPE "Debug"
#endif





int main() {

    //set current path
    std::filesystem::path current_path = std::filesystem::current_path();
    std::string path;
    try {
        current_path = current_path.parent_path().parent_path();
//        path = current_path.string() + "/app";
        path = current_path.string() ;
        std::filesystem::current_path(path);
        std::cout << "Curent path  set to: " << path << std::endl;
    }
    catch (const std::filesystem::filesystem_error &ex) {
        std::cout << "Error setting current path:  " << ex.what() << std::endl;
    }

    std::cout << "Begin" << std::endl;
    std::string onnxModelPath = "newmodel.onnx";


    Options options;
    options.precision = Precision::FP32;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;

    Engine engine(options);

    if(!engine.buildLoadNetwork(onnxModelPath)){
        std::cout << "Engine build failed" << std::endl;
    }


//    Model supports dynamic batch size
//    Input name: image_input
//    Input dims: -1 224 224 3
//    Input name: tabular_input
//    Input dims: -1 182


//image
//    cv::cuda::GpuMat image(224, 224, CV_32FC3);
//

//    /home/toor/Code/Parking/821263542.png

//    //opencv read image and convert to GpuMat
    cv::Mat img = cv::imread("/home/toor/Code/Parking/LAST_DATA/");
    cv::cuda::GpuMat image;
    cv::cvtColor(image,image,cv::COLOR_BGR2RGB);
    image.upload(img);

    std::vector<float> lidarData(182, 0.0f);
    lidarData = {

    };




    while (true){
        std::this_thread::sleep_for(std::chrono::seconds(1));
        // Prepare input data (image and distances)

        // Run inference
        try {
            engine.RunInference(image, lidarData);
        } catch (const std::exception& e) {
            std::cerr << "Error running inference: " << e.what() << std::endl;
            return 1;
        }
    }





}
