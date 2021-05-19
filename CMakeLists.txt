cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(CNQS LANGUAGES CXX VERSION 1.0.0)

option(CNQS_BUILD_PDESOLVER "Build PDE eigensolver subpackage" ON)
option(CNQS_BUILD_VMCSOLVER "Build VMC eigensolver subpackage" ON)

find_package(yaml-cpp REQUIRED)

if(CNQS_BUILD_PDESOLVER)
    find_package(Trilinos REQUIRED COMPONENTS Teuchos Tpetra Belos)

    set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS ${Trilinos_CXX_COMPILER_FLAGS}
                        ${CMAKE_CXX_FLAGS})
endif()

if(CNQS_BUILD_VMCSOLVER)
    find_package(Boost REQUIRED COMPONENTS math_tr1)
    find_package(blaspp REQUIRED)
    find_package(lapackpp REQUIRED)
endif()

add_subdirectory(src)

add_subdirectory(app)

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(CNQS_ENABLE_DOCS "Build the package documentation" ON)
endif()

if(CNQS_ENABLE_DOCS)
    add_subdirectory(doc)
endif()

configure_file(hamiltonian.yaml hamiltonian.yaml COPYONLY)
configure_file(config.yaml config.yaml COPYONLY)
