add_library(cnqs cnqs_state.cpp
                 cnqs_hamiltonian.cpp)
target_include_directories(cnqs PUBLIC ${PROJECT_SOURCE_DIR}/include
                                       ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(cnqs PUBLIC ${ARMADILLO_LIBRARIES})
target_compile_features(cnqs PUBLIC cxx_std_11)

add_library(cnqs::cnqs ALIAS cnqs)
