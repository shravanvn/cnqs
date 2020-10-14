# cNQS PDE Solver

## How to Build?

This library uses the [CMake](http://www.cmake.org/) to support compilation.
The basic build steps are outlined below:

1.  To compile the programs, the following prerequisites are required:

    -   [nlohmann_json](https://github.com/nlohmann/json)

    -   [Trilinos](https://github.com/trilinos/Trilinos)

    We highly recommend using the [spack](https://spack.io/) package manager to
    install both - this will automatically take care of any secondary
    dependencies. We have tested the code with `nlohmann-json@3.7.2` and
    `trilinos@12.18.1`.

2.  Separate out the build tree: from inside the current directory, run

    ```sh
    mkdir build
    cd build
    ```

3.  To generate build configuration, run

    ```sh
    cmake \
        -D nlohmann_json_PREFIX=<NLOHMANN_JSON_ROOT> \
        -D Trilinos_PREFIX=<TRILINOS_ROOT> \
        ..
    ```

4.  To build the library and programs, run

    ```sh
    make
    ```

    Inside the `app` directory, you will find the `basic_problem` and
    `fourier_problem` programs. Run

    ```sh
    cd app
    basic_problem -h
    fourier_problem -h
    ```

    to get a basic idea on how to set run parameters.
