macro(CNQS_add_app APPNAME)
    add_executable(${APPNAME} ${ARGN})
    target_link_libraries(${APPNAME} CNQS::cnqs)
endmacro()

CNQS_add_app(basic_problem BasicProblem.cpp)
CNQS_add_app(fourier_problem FourierProblem.cpp)

configure_file(network.yaml network.yaml COPYONLY)
