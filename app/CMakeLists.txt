macro(cnqs_add_app APPNAME)
    add_executable(${APPNAME} ${ARGN})
    target_link_libraries(${APPNAME} cnqs::cnqs)
endmacro()

cnqs_add_app(inverse_power_direct inverse_power_direct.cpp)
cnqs_add_app(inverse_power_fourier inverse_power_fourier.cpp)
