# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/toor/Code/Parking

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/toor/Code/Parking/build

# Include any dependencies generated for this target.
include app/CMakeFiles/executable.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include app/CMakeFiles/executable.dir/compiler_depend.make

# Include the progress variables for this target.
include app/CMakeFiles/executable.dir/progress.make

# Include the compile flags for this target's objects.
include app/CMakeFiles/executable.dir/flags.make

app/CMakeFiles/executable.dir/main.cpp.o: app/CMakeFiles/executable.dir/flags.make
app/CMakeFiles/executable.dir/main.cpp.o: /home/toor/Code/Parking/app/main.cpp
app/CMakeFiles/executable.dir/main.cpp.o: app/CMakeFiles/executable.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/toor/Code/Parking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object app/CMakeFiles/executable.dir/main.cpp.o"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT app/CMakeFiles/executable.dir/main.cpp.o -MF CMakeFiles/executable.dir/main.cpp.o.d -o CMakeFiles/executable.dir/main.cpp.o -c /home/toor/Code/Parking/app/main.cpp

app/CMakeFiles/executable.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/executable.dir/main.cpp.i"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/toor/Code/Parking/app/main.cpp > CMakeFiles/executable.dir/main.cpp.i

app/CMakeFiles/executable.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/executable.dir/main.cpp.s"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/toor/Code/Parking/app/main.cpp -o CMakeFiles/executable.dir/main.cpp.s

app/CMakeFiles/executable.dir/ai.cpp.o: app/CMakeFiles/executable.dir/flags.make
app/CMakeFiles/executable.dir/ai.cpp.o: /home/toor/Code/Parking/app/ai.cpp
app/CMakeFiles/executable.dir/ai.cpp.o: app/CMakeFiles/executable.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/toor/Code/Parking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object app/CMakeFiles/executable.dir/ai.cpp.o"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT app/CMakeFiles/executable.dir/ai.cpp.o -MF CMakeFiles/executable.dir/ai.cpp.o.d -o CMakeFiles/executable.dir/ai.cpp.o -c /home/toor/Code/Parking/app/ai.cpp

app/CMakeFiles/executable.dir/ai.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/executable.dir/ai.cpp.i"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/toor/Code/Parking/app/ai.cpp > CMakeFiles/executable.dir/ai.cpp.i

app/CMakeFiles/executable.dir/ai.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/executable.dir/ai.cpp.s"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/toor/Code/Parking/app/ai.cpp -o CMakeFiles/executable.dir/ai.cpp.s

app/CMakeFiles/executable.dir/engine.cpp.o: app/CMakeFiles/executable.dir/flags.make
app/CMakeFiles/executable.dir/engine.cpp.o: /home/toor/Code/Parking/app/engine.cpp
app/CMakeFiles/executable.dir/engine.cpp.o: app/CMakeFiles/executable.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/toor/Code/Parking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object app/CMakeFiles/executable.dir/engine.cpp.o"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT app/CMakeFiles/executable.dir/engine.cpp.o -MF CMakeFiles/executable.dir/engine.cpp.o.d -o CMakeFiles/executable.dir/engine.cpp.o -c /home/toor/Code/Parking/app/engine.cpp

app/CMakeFiles/executable.dir/engine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/executable.dir/engine.cpp.i"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/toor/Code/Parking/app/engine.cpp > CMakeFiles/executable.dir/engine.cpp.i

app/CMakeFiles/executable.dir/engine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/executable.dir/engine.cpp.s"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/toor/Code/Parking/app/engine.cpp -o CMakeFiles/executable.dir/engine.cpp.s

app/CMakeFiles/executable.dir/pch.cpp.o: app/CMakeFiles/executable.dir/flags.make
app/CMakeFiles/executable.dir/pch.cpp.o: /home/toor/Code/Parking/app/pch.cpp
app/CMakeFiles/executable.dir/pch.cpp.o: app/CMakeFiles/executable.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/toor/Code/Parking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object app/CMakeFiles/executable.dir/pch.cpp.o"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT app/CMakeFiles/executable.dir/pch.cpp.o -MF CMakeFiles/executable.dir/pch.cpp.o.d -o CMakeFiles/executable.dir/pch.cpp.o -c /home/toor/Code/Parking/app/pch.cpp

app/CMakeFiles/executable.dir/pch.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/executable.dir/pch.cpp.i"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/toor/Code/Parking/app/pch.cpp > CMakeFiles/executable.dir/pch.cpp.i

app/CMakeFiles/executable.dir/pch.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/executable.dir/pch.cpp.s"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/toor/Code/Parking/app/pch.cpp -o CMakeFiles/executable.dir/pch.cpp.s

app/CMakeFiles/executable.dir/Utils.cpp.o: app/CMakeFiles/executable.dir/flags.make
app/CMakeFiles/executable.dir/Utils.cpp.o: /home/toor/Code/Parking/app/Utils.cpp
app/CMakeFiles/executable.dir/Utils.cpp.o: app/CMakeFiles/executable.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/toor/Code/Parking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object app/CMakeFiles/executable.dir/Utils.cpp.o"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT app/CMakeFiles/executable.dir/Utils.cpp.o -MF CMakeFiles/executable.dir/Utils.cpp.o.d -o CMakeFiles/executable.dir/Utils.cpp.o -c /home/toor/Code/Parking/app/Utils.cpp

app/CMakeFiles/executable.dir/Utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/executable.dir/Utils.cpp.i"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/toor/Code/Parking/app/Utils.cpp > CMakeFiles/executable.dir/Utils.cpp.i

app/CMakeFiles/executable.dir/Utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/executable.dir/Utils.cpp.s"
	cd /home/toor/Code/Parking/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/toor/Code/Parking/app/Utils.cpp -o CMakeFiles/executable.dir/Utils.cpp.s

# Object files for target executable
executable_OBJECTS = \
"CMakeFiles/executable.dir/main.cpp.o" \
"CMakeFiles/executable.dir/ai.cpp.o" \
"CMakeFiles/executable.dir/engine.cpp.o" \
"CMakeFiles/executable.dir/pch.cpp.o" \
"CMakeFiles/executable.dir/Utils.cpp.o"

# External object files for target executable
executable_EXTERNAL_OBJECTS =

app/executable: app/CMakeFiles/executable.dir/main.cpp.o
app/executable: app/CMakeFiles/executable.dir/ai.cpp.o
app/executable: app/CMakeFiles/executable.dir/engine.cpp.o
app/executable: app/CMakeFiles/executable.dir/pch.cpp.o
app/executable: app/CMakeFiles/executable.dir/Utils.cpp.o
app/executable: app/CMakeFiles/executable.dir/build.make
app/executable: /usr/local/lib/libopencv_gapi.so.4.9.0
app/executable: /usr/local/lib/libopencv_stitching.so.4.9.0
app/executable: /usr/local/lib/libopencv_aruco.so.4.9.0
app/executable: /usr/local/lib/libopencv_bgsegm.so.4.9.0
app/executable: /usr/local/lib/libopencv_bioinspired.so.4.9.0
app/executable: /usr/local/lib/libopencv_ccalib.so.4.9.0
app/executable: /usr/local/lib/libopencv_cudabgsegm.so.4.9.0
app/executable: /usr/local/lib/libopencv_cudafeatures2d.so.4.9.0
app/executable: /usr/local/lib/libopencv_cudaobjdetect.so.4.9.0
app/executable: /usr/local/lib/libopencv_cudastereo.so.4.9.0
app/executable: /usr/local/lib/libopencv_dnn_objdetect.so.4.9.0
app/executable: /usr/local/lib/libopencv_dnn_superres.so.4.9.0
app/executable: /usr/local/lib/libopencv_dpm.so.4.9.0
app/executable: /usr/local/lib/libopencv_face.so.4.9.0
app/executable: /usr/local/lib/libopencv_freetype.so.4.9.0
app/executable: /usr/local/lib/libopencv_fuzzy.so.4.9.0
app/executable: /usr/local/lib/libopencv_hfs.so.4.9.0
app/executable: /usr/local/lib/libopencv_img_hash.so.4.9.0
app/executable: /usr/local/lib/libopencv_intensity_transform.so.4.9.0
app/executable: /usr/local/lib/libopencv_line_descriptor.so.4.9.0
app/executable: /usr/local/lib/libopencv_mcc.so.4.9.0
app/executable: /usr/local/lib/libopencv_quality.so.4.9.0
app/executable: /usr/local/lib/libopencv_rapid.so.4.9.0
app/executable: /usr/local/lib/libopencv_reg.so.4.9.0
app/executable: /usr/local/lib/libopencv_rgbd.so.4.9.0
app/executable: /usr/local/lib/libopencv_saliency.so.4.9.0
app/executable: /usr/local/lib/libopencv_signal.so.4.9.0
app/executable: /usr/local/lib/libopencv_stereo.so.4.9.0
app/executable: /usr/local/lib/libopencv_structured_light.so.4.9.0
app/executable: /usr/local/lib/libopencv_superres.so.4.9.0
app/executable: /usr/local/lib/libopencv_surface_matching.so.4.9.0
app/executable: /usr/local/lib/libopencv_tracking.so.4.9.0
app/executable: /usr/local/lib/libopencv_videostab.so.4.9.0
app/executable: /usr/local/lib/libopencv_wechat_qrcode.so.4.9.0
app/executable: /usr/local/lib/libopencv_xfeatures2d.so.4.9.0
app/executable: /usr/local/lib/libopencv_xobjdetect.so.4.9.0
app/executable: /usr/local/lib/libopencv_xphoto.so.4.9.0
app/executable: /opt/cuda/lib64/libcudart_static.a
app/executable: /usr/lib/librt.a
app/executable: /usr/lib/libnvinfer.so
app/executable: /usr/lib/libnvonnxparser.so
app/executable: /usr/lib/libnvonnxparser.so
app/executable: libs/yaml-cpp/libyaml-cpp.a
app/executable: /usr/local/lib/libopencv_shape.so.4.9.0
app/executable: /usr/local/lib/libopencv_highgui.so.4.9.0
app/executable: /usr/local/lib/libopencv_datasets.so.4.9.0
app/executable: /usr/local/lib/libopencv_plot.so.4.9.0
app/executable: /usr/local/lib/libopencv_text.so.4.9.0
app/executable: /usr/local/lib/libopencv_ml.so.4.9.0
app/executable: /usr/local/lib/libopencv_phase_unwrapping.so.4.9.0
app/executable: /usr/local/lib/libopencv_cudacodec.so.4.9.0
app/executable: /usr/local/lib/libopencv_videoio.so.4.9.0
app/executable: /usr/local/lib/libopencv_cudaoptflow.so.4.9.0
app/executable: /usr/local/lib/libopencv_cudalegacy.so.4.9.0
app/executable: /usr/local/lib/libopencv_cudawarping.so.4.9.0
app/executable: /usr/local/lib/libopencv_optflow.so.4.9.0
app/executable: /usr/local/lib/libopencv_ximgproc.so.4.9.0
app/executable: /usr/local/lib/libopencv_video.so.4.9.0
app/executable: /usr/local/lib/libopencv_imgcodecs.so.4.9.0
app/executable: /usr/local/lib/libopencv_objdetect.so.4.9.0
app/executable: /usr/local/lib/libopencv_calib3d.so.4.9.0
app/executable: /usr/local/lib/libopencv_dnn.so.4.9.0
app/executable: /usr/local/lib/libopencv_features2d.so.4.9.0
app/executable: /usr/local/lib/libopencv_flann.so.4.9.0
app/executable: /usr/local/lib/libopencv_photo.so.4.9.0
app/executable: /usr/local/lib/libopencv_cudaimgproc.so.4.9.0
app/executable: /usr/local/lib/libopencv_cudafilters.so.4.9.0
app/executable: /usr/local/lib/libopencv_imgproc.so.4.9.0
app/executable: /usr/local/lib/libopencv_cudaarithm.so.4.9.0
app/executable: /usr/local/lib/libopencv_core.so.4.9.0
app/executable: /usr/local/lib/libopencv_cudev.so.4.9.0
app/executable: app/CMakeFiles/executable.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/toor/Code/Parking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable executable"
	cd /home/toor/Code/Parking/build/app && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/executable.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
app/CMakeFiles/executable.dir/build: app/executable
.PHONY : app/CMakeFiles/executable.dir/build

app/CMakeFiles/executable.dir/clean:
	cd /home/toor/Code/Parking/build/app && $(CMAKE_COMMAND) -P CMakeFiles/executable.dir/cmake_clean.cmake
.PHONY : app/CMakeFiles/executable.dir/clean

app/CMakeFiles/executable.dir/depend:
	cd /home/toor/Code/Parking/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/toor/Code/Parking /home/toor/Code/Parking/app /home/toor/Code/Parking/build /home/toor/Code/Parking/build/app /home/toor/Code/Parking/build/app/CMakeFiles/executable.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : app/CMakeFiles/executable.dir/depend

