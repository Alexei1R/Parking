# CMAKE generated file: DO NOT EDIT!
# Generated by CMake Version 3.29
cmake_policy(SET CMP0009 NEW)

# yaml-cpp-sources at libs/yaml-cpp/CMakeLists.txt:72 (file)
file(GLOB NEW_GLOB LIST_DIRECTORIES true "/home/toor/Code/Parking/libs/yaml-cpp/src/*.cpp")
set(OLD_GLOB
  "/home/toor/Code/Parking/libs/yaml-cpp/src/binary.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/convert.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/depthguard.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/directives.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/emit.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/emitfromevents.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/emitter.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/emitterstate.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/emitterutils.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/exceptions.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/exp.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/memory.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/node.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/node_data.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/nodebuilder.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/nodeevents.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/null.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/ostream_wrapper.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/parse.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/parser.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/regex_yaml.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/scanner.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/scanscalar.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/scantag.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/scantoken.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/simplekey.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/singledocparser.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/stream.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/tag.cpp"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/toor/Code/Parking/build/CMakeFiles/cmake.verify_globs")
endif()

# yaml-cpp-contrib-sources at libs/yaml-cpp/CMakeLists.txt:71 (file)
file(GLOB NEW_GLOB LIST_DIRECTORIES true "/home/toor/Code/Parking/libs/yaml-cpp/src/contrib/*.cpp")
set(OLD_GLOB
  "/home/toor/Code/Parking/libs/yaml-cpp/src/contrib/graphbuilder.cpp"
  "/home/toor/Code/Parking/libs/yaml-cpp/src/contrib/graphbuilderadapter.cpp"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/toor/Code/Parking/build/CMakeFiles/cmake.verify_globs")
endif()