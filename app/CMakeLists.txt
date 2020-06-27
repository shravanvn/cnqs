macro(cnqs_add_app APPNAME)
    add_executable(${APPNAME} ${ARGN})
    target_link_libraries(${APPNAME} cnqs::cnqs)
endmacro()

configure_file(parameters_basic.txt parameters_basic.txt COPYONLY)
cnqs_add_app(inv_power_iter_basic InvPowerIterBasic.cpp)
