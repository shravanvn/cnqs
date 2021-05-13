add_executable(vmcsolver
    main.cpp
)
target_link_libraries(vmcsolver PRIVATE
    cnqs::cnqs
)
