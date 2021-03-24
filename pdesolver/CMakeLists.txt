cmake_minimum_required(VERSION 3.11)
project(CNQS LANGUAGES CXX VERSION 0.2.0)

set(CMAKE_PREFIX_PATH ${YAML_CPP_PREFIX}
                      ${TRILINOS_PREFIX}
                      ${CMAKE_PREFIX_PATH})

find_package(yaml-cpp REQUIRED)
find_package(Trilinos REQUIRED COMPONENTS Teuchos Tpetra Belos)

set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
set(CMAKE_CXX_FLAGS ${Trilinos_CXX_COMPILER_FLAGS}
                    ${CMAKE_CXX_FLAGS})

add_library(cnqs INTERFACE)
target_include_directories(cnqs INTERFACE ${PROJECT_SOURCE_DIR}/include
                                          ${Trilinos_INCLUDE_DIRS}
                                          ${Trilinos_TPL_INCLUDE_DIRS})
target_link_libraries(cnqs INTERFACE yaml-cpp
                                     ${Trilinos_LIBRARIES}
                                     ${Trilinos_TPL_LIBRARIES})
target_compile_features(cnqs INTERFACE cxx_std_11)
add_library(CNQS::cnqs ALIAS cnqs)

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
