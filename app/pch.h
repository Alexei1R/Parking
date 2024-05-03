//
// Created by toor on 11/7/23.
//

#ifndef ATOM_ATOMPCH_H
#define ATOM_ATOMPCH_H

#include <iostream>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>

#include <string>
#include <sstream>
#include <array>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <string>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <map>
#include <cctype>
#include <set>


#include <cmath>

#ifdef _WIN32
#include <windows.h> // Ug, for NukeProcess -- see below
#else

#include <unistd.h>
#include <signal.h>

#endif

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



typedef struct {
    float pixelFromCenter;
    float confidence;
    std::string label;


} Sign;


#endif //ATOM_ATOMPCH_H
