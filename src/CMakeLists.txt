add_library(cnqs Cnqs_Network.cpp
                 Cnqs_BasicProblem.cpp
                 Cnqs_FourierProblem.cpp)

target_include_directories(cnqs PUBLIC ${PROJECT_SOURCE_DIR}/include
                                       ${Trilinos_INCLUDE_DIRS}
                                       ${Trilinos_TPL_INCLUDE_DIRS})

target_link_libraries(cnqs PUBLIC nlohmann_json::nlohmann_json
                                  ${Trilinos_LIBRARIES}
                                  ${Trilinos_TPL_LIBRARIES}
                                  OpenMP::OpenMP_CXX
                                  MPI::MPI_CXX)

target_compile_features(cnqs PUBLIC cxx_std_11)

add_library(cnqs::cnqs ALIAS cnqs)
