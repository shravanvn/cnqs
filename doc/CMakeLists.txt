find_package(Doxygen)

if(DOXYGEN_FOUND)
    configure_file(Doxyfile Doxyfile @ONLY)

    add_custom_target(docs ALL
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating documentation"
    )
else()
    message(WARNING "Could not find Doxygen. Skipping documentation generation")
endif()
