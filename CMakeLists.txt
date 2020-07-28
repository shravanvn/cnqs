cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)
project(CNQS LANGUAGES CXX VERSION 0.2.0)

find_package(nlohmann_json REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Trilinos REQUIRED COMPONENTS Teuchos Tpetra Belos)

add_library(cnqs INTERFACE)
target_include_directories(cnqs INTERFACE ${PROJECT_SOURCE_DIR}/include
                                          ${Trilinos_INCLUDE_DIRS}
                                          ${Trilinos_TPL_INCLUDE_DIRS})
target_link_libraries(cnqs INTERFACE nlohmann_json::nlohmann_json
                                     ${Trilinos_LIBRARIES}
                                     ${Trilinos_TPL_LIBRARIES}
                                     OpenMP::OpenMP_CXX
                                     MPI::MPI_CXX)
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
