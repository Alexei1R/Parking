#!/bin/bash
build_and_run() {
    if [ ! -d "build" ]; then
        mkdir build
    fi
    cd build || exit
    cmake ..
    make -j"$(nproc)"
    cd ./app/

   ./executable
}

notify-send "ATOM" "Begin build."
build_and_run
