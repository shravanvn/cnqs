#! /bin/bash

#mkdir build # out of source build
cd build
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER=/usr/bin/g++ \
    -D CNQS_BUILD_PDESOLVER=OFF \
    -D CNQS_BUILD_VMCSOLVER=ON \
    -D CNQS_BUILD_DOCS=ON \
    -D yaml-cpp_ROOT=/usr/local/include/yaml-cpp \
    -D Boost_ROOT=/usr/share/doc/libboost-all-dev \
    -D blaspp_ROOT=/home/yabin/Software/blaspp/build \
    -D lapackpp_ROOT=/home/yabin/Software/lapackpp/build \
    ..
make

#     -D Trilinos_ROOT=/path/to/trilinos/install/prefix \
