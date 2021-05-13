cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(cnqs LANGUAGES CXX)

find_package(Boost REQUIRED COMPONENTS math_tr1)
find_package(blaspp REQUIRED)
find_package(lapackpp REQUIRED)
find_package(yaml-cpp REQUIRED)

add_subdirectory(src)
add_subdirectory(app)

configure_file(hamiltonian.yaml hamiltonian.yaml COPYONLY)
configure_file(config.yaml config.yaml COPYONLY)
