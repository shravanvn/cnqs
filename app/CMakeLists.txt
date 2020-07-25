macro(CNQS_add_app APPNAME)
    add_executable(${APPNAME} ${ARGN})
    target_link_libraries(${APPNAME} cnqs::cnqs)
endmacro()

CNQS_add_app(basic_problem BasicProblem.cpp)
CNQS_add_app(fourier_problem FourierProblem.cpp)

configure_file(network_square.json network_square.json COPYONLY)
