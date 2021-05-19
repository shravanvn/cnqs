if(CNQS_BUILD_PDESOLVER)
    add_executable(pdesolver_fd pdesolver_fd.cpp)
    target_link_libraries(pdesolver_fd PRIVATE CNQS::pdesolver)

    add_executable(pdesolver_sd pdesolver_sd.cpp)
    target_link_libraries(pdesolver_sd PRIVATE CNQS::pdesolver)
endif()

if(CNQS_BUILD_VMCSOLVER)
    add_executable(vmcsolver vmcsolver.cpp)
    target_link_libraries(vmcsolver PRIVATE CNQS::vmcsolver)
endif()
