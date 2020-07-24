macro(CNQS_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} cnqs::cnqs)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endmacro()

configure_file(network.json network.json COPYONLY)

CNQS_add_test(cnqs_network_test Cnqs_NetworkTest.cpp)
CNQS_add_test(cnqs_basic_problem_test Cnqs_BasicProblemTest.cpp)
CNQS_add_test(cnqs_fourier_problem_test Cnqs_FourierProblemTest.cpp)
