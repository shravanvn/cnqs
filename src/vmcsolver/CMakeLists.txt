add_library(cnqs_vmcsolver
    config.cpp
    nqs.cpp
    optimization.cpp
    sampler.cpp
)
target_include_directories(cnqs_vmcsolver PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(cnqs_vmcsolver PRIVATE
    Boost::math_tr1
    Boost::filesystem
    blaspp
    lapackpp
    yaml-cpp
)
target_compile_features(cnqs_vmcsolver PUBLIC cxx_std_11)

add_library(CNQS::vmcsolver ALIAS cnqs_vmcsolver)
