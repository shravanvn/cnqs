add_library(cnqs
    config.cpp
    nqs.cpp
    optimization.cpp
    sampler.cpp
)
target_include_directories(cnqs PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(cnqs PRIVATE
    Boost::math_tr1
    blaspp
    lapackpp
    yaml-cpp
)
target_compile_features(cnqs PUBLIC
    cxx_std_11
)

add_library(cnqs::cnqs ALIAS cnqs)
