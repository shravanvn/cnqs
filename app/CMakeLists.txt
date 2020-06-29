macro(CNQS_add_app APPNAME)
    add_executable(${APPNAME} ${ARGN})
    target_link_libraries(${APPNAME} cnqs::cnqs)
endmacro()

configure_file(parameters_basic.txt parameters_basic.txt COPYONLY)
CNQS_add_app(inv_power_iter_basic InvPowerIterBasic.cpp)

configure_file(parameters_fourier.txt parameters_fourier.txt COPYONLY)
CNQS_add_app(inv_power_iter_fourier InvPowerIterFourier.cpp)
