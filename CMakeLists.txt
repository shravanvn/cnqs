cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)
project(CNQS LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

add_subdirectory(src)

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(cnqs_ENABLE_TESTS "Build the package tests" ON)
endif()

if(cnqs_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

add_subdirectory(app)

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(cnqs_ENABLE_DOCS "Build the package documentation" ON)
endif()

if(cnqs_ENABLE_DOCS)
    add_subdirectory(docs)
endif()
