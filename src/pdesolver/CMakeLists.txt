add_library(cnqs_pdesolver INTERFACE)
target_include_directories(cnqs_pdesolver INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${Trilinos_INCLUDE_DIRS}
    ${Trilinos_TPL_INCLUDE_DIRS}
)
target_link_libraries(cnqs_pdesolver INTERFACE
    yaml-cpp
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
)
target_compile_features(cnqs_pdesolver INTERFACE cxx_std_11)

add_library(CNQS::pdesolver ALIAS cnqs_pdesolver)
