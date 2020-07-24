if(NOT TARGET gtest)
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
endif()

macro(CNQS_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} cnqs::cnqs gtest)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endmacro()

CNQS_add_test(cnqs_vector_test CnqsVectorTest.cpp)
CNQS_add_test(cnqs_basic_operator_test CnqsBasicOperatorTest.cpp)
CNQS_add_test(cnqs_fourier_operator_test CnqsFourierOperatorTest.cpp)
CNQS_add_test(cnqs_trivial_preconditioner_test CnqsTrivialPreconditionerTest.cpp)
CNQS_add_test(cnqs_fourier_preconditioner_test CnqsFourierPreconditionerTest.cpp)
CNQS_add_test(inv_power_iter_test InvPowerIterTest.cpp)

CNQS_add_test(cnqs_network_test Cnqs_NetworkTest.cpp)
CNQS_add_test(cnqs_basic_problem_test Cnqs_BasicProblemTest.cpp)
CNQS_add_test(cnqs_fourier_problem_test Cnqs_FourierProblemTest.cpp)
