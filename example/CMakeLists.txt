macro(CNQS_add_example APPNAME)
    add_executable(${APPNAME} ${ARGN})
    target_link_libraries(${APPNAME} cnqs::cnqs)
endmacro()

configure_file(network.json network.json COPYONLY)

CNQS_add_example(basic_problem BasicProblem.cpp)
CNQS_add_example(fourier_problem FourierProblem.cpp)
