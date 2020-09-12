find_package(Doxygen)

if(DOXYGEN_FOUND)
    get_target_property(CNQS_PUBLIC_HEADER_DIR CNQS::cnqs INTERFACE_INCLUDE_DIRECTORIES)
    file(GLOB_RECURSE CNQS_PUBLIC_HEADERS ${CNQS_PUBLIC_HEADER_DIR}/*.hpp)

    configure_file(Doxyfile.in Doxyfile @ONLY)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
                       DEPENDS ${CNQS_PUBLIC_HEADERS}
                       COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
                       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                       MAIN_DEPENDENCY Doxyfile
                       COMMENT "Generating documentation")

    add_custom_target(docs ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/html/index.html)
else()
    message(WARNING "Could not find Doxygen. Skipping documentation generation")
endif()
