if(CNQS_BUILD_PDESOLVER)
    add_executable(pdesolver_fd pdesolver_fd.cpp)
    target_link_libraries(pdesolver_fd PRIVATE CNQS::pdesolver)
    target_compile_features(pdesolver_fd PRIVATE cxx_std_11)

    add_executable(pdesolver_sd pdesolver_sd.cpp)
    target_link_libraries(pdesolver_sd PRIVATE CNQS::pdesolver)
    target_compile_features(pdesolver_sd PRIVATE cxx_std_11)

    add_executable(pdesolver_tt pdesolver_tt.cpp)
    target_link_libraries(pdesolver_tt PRIVATE CNQS::pdesolver)
    target_compile_features(pdesolver_tt PRIVATE cxx_std_11)
endif()

if(CNQS_BUILD_VMCSOLVER)
    add_executable(vmcsolver vmcsolver.cpp)
    target_link_libraries(vmcsolver PRIVATE CNQS::vmcsolver)
    target_compile_features(vmcsolver PRIVATE cxx_std_11)
endif()
