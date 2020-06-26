add_subdirectory("${PROJECT_SOURCE_DIR}/extern/iml" "extern/iml")

add_library(cnqs cnqs_state.cpp
                 cnqs_hamiltonian.cpp
                 cnqs_hamiltonian_direct.cpp
                 utils.cpp)

                 target_include_directories(cnqs PUBLIC ${PROJECT_SOURCE_DIR}/include
                                       ${HDF5_INCLUDE_DIRS})
target_include_directories(cnqs PRIVATE ${HDF5_INCLUDE_DIRS})

target_link_libraries(cnqs PUBLIC ${HDF5_LIBRARIES})
target_link_libraries(cnqs PRIVATE iml::iml)

target_compile_features(cnqs PUBLIC cxx_std_11)

add_library(cnqs::cnqs ALIAS cnqs)
