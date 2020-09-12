# CNQS

Fast solvers for computing continuous-variable neural quantum states.

## PDE-Solver Dependencies

-   [Trilinos](https://github.com/Trilinos/Trilinos) with at least the following
    packages enabled: `Teuchos`, `Tpetra` and `Belos`, and built with MPI and
    OpenMP support. Code has been tested with version 12.18.1.

-   [nlohmann_json](https://github.com/nlohmann/json). Code has been tested with
    version 3.8.0.

## HMC-Solver Dependencies

-   [Conda](https://docs.conda.io/) environment specified in the `hmcsolver`
    subdirectory.
