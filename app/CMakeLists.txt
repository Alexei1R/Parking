cmake_minimum_required(VERSION 3.0)
project(ATOM)
find_package(OpenCV REQUIRED)


message(==============${CMAKE_CURRENT_SOURCE_DIR})


if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(Link to TensorRT)
    message(${CMAKE_CURRENT_SOURCE_DIR})
    #strip before last /
    string(REGEX REPLACE "/[^/]*$" "" MAIN_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    message(${MAIN_CMAKE_PATH})

    include(${MAIN_CMAKE_PATH}/cmake/ccache.cmake)
    set(CMAKE_MODULE_PATH "${MAIN_CMAKE_PATH}/cmake" ${MAIN_CMAKE_PATH})
    set(TensorRT_DIR /usr/)
    set(TensorRT_NVPARSERS_LIBRARY /usr/lib/libnvonnxparser.so)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
    find_package(TensorRT REQUIRED COMPONENTS nvparsers)
    find_package(CUDA REQUIRED)
    find_package(OpenCV REQUIRED)
endif ()


set(SOURCES
        main.cpp
        engine.cpp
        engine.h
        pch.cpp
        pch.h
        Utils.cpp
        Utils.h
)


add_executable(executable ${SOURCES})


if (CMAKE_BUILD_TYPE STREQUAL "Release")

    target_link_libraries(executable
            ${OpenCV_LIBS}
            ${CUDA_LIBRARIES}
            ${CMAKE_THREAD_LIBS_INIT}
            ${TensorRT_LIBRARIES}
            yaml-cpp::yaml-cpp

    )

    target_include_directories(executable PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})

else ()
    target_link_libraries(executable
            ${OpenCV_LIBS}
            yaml-cpp::yaml-cpp
    )
    target_include_directories(executable PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)

endif ()

