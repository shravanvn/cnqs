add_library(cnqs_pdesolver
    hamiltonian.cpp
    finite_difference_problem.cpp
    spectral_difference_problem.cpp
)
target_include_directories(cnqs_pdesolver PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${Trilinos_INCLUDE_DIRS}
    ${Trilinos_TPL_INCLUDE_DIRS}
)
target_link_libraries(cnqs_pdesolver PUBLIC
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
)
target_link_libraries(cnqs_pdesolver PRIVATE yaml-cpp)
target_compile_features(cnqs_pdesolver PUBLIC cxx_std_11)

add_library(CNQS::pdesolver ALIAS cnqs_pdesolver)
