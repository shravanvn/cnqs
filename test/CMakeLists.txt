add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")

mark_as_advanced(
    BUILD_SHARED_LIBS
    BUILD_GTEST
    BUILD_GMOCK
    gtest_build_samples
    gtest_build_tests
    gtest_disable_pthreads
    gtest_force_shared_crt
    gtest_hide_internal_symbols
    gmock_build_tests
)

set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)
set_target_properties(gmock PROPERTIES FOLDER extern)
set_target_properties(gmock_main PROPERTIES FOLDER extern)

macro(cnqs_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} cnqs::cnqs gmock gtest gmock_main gtest_main)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endmacro()

cnqs_add_test(cnqs_state_test cnqs_state_test.cpp)
cnqs_add_test(cnqs_hamiltonian_direct_test cnqs_hamiltonian_direct_test.cpp)
cnqs_add_test(cnqs_hamiltonian_fourier_test cnqs_hamiltonian_fourier_test.cpp)
