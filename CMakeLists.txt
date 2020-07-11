cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)
project(CNQS LANGUAGES CXX VERSION 0.1.0)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Trilinos REQUIRED)

add_subdirectory(src)

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(CNQS_ENABLE_TESTS "Build the package tests" ON)
endif()

if(CNQS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

add_subdirectory(app)

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(CNQS_ENABLE_DOCS "Build the package documentation" ON)
endif()

if(CNQS_ENABLE_DOCS)
    add_subdirectory(docs)
endif()
