cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)
project(cnqs LANGUAGES C CXX VERSION 0.1.0)

add_subdirectory(src)

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(cnqs_ENABLE_TESTS "Build the package tests" ON)
endif()

if(cnqs_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

