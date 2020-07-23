add_subdirectory("${PROJECT_SOURCE_DIR}/extern/iml" "extern/iml")

add_library(cnqs CnqsVector.cpp
                 CnqsOperator.cpp
                 CnqsBasicOperator.cpp
                 CnqsFourierOperator.cpp
                 CnqsPreconditioner.cpp
                 CnqsTrivialPreconditioner.cpp
                 CnqsFourierPreconditioner.cpp
                 InvPowerIter.cpp
                 Cnqs_Network.cpp
                 Cnqs_Problem.cpp
                 Cnqs_BasicProblem.cpp
                 Utils.cpp)

target_include_directories(cnqs PUBLIC ${PROJECT_SOURCE_DIR}/include
                                       ${Trilinos_INCLUDE_DIRS}
                                       ${Trilinos_TPL_INCLUDE_DIRS})

target_link_libraries(cnqs PUBLIC ${Trilinos_LIBRARIES}
                                  ${Trilinos_TPL_LIBRARIES}
                                  OpenMP::OpenMP_CXX
                                  MPI::MPI_CXX)
target_link_libraries(cnqs PRIVATE iml::iml)

target_compile_features(cnqs PUBLIC cxx_std_11)

add_library(cnqs::cnqs ALIAS cnqs)
