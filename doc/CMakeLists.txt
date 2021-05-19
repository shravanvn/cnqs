find_package(Doxygen)

if(DOXYGEN_FOUND)
    configure_file(Doxyfile Doxyfile @ONLY)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        MAIN_DEPENDENCY Doxyfile
        COMMENT "Generating documentation"
    )

    add_custom_target(docs ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
    )
else()
    message(WARNING "Could not find Doxygen. Skipping documentation generation")
endif()
