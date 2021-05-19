# CNQS

C++ implementations for eigensolvers for quantum rotor Hamiltonians.

## Installing Dependencies

To build the libraries/executables in this project, the following are required:

*   C++11 compatible compiler
*   [CMake](http://cmake.org/) >= 3.12.0
*   [yaml-cpp](https://github.com/jbeder/yaml-cpp)
*   If building the PDE eigensolvers:
    *   [Trilinos](https://github.com/trilinos/Trilinos/)
*   If building the VMC eigensolver:
    *   [Boost](https://www.boost.org/) (only the math_tr1 sub-library is
        sufficient)
    *   [BLAS++](https://bitbucket.org/icl/blaspp/)
    *   [LAPACK++](https://bitbucket.org/icl/lapackpp/)

The code is tested on macOS Big Sur with

*   GCC 9.3.0, installed with Homebrew
*   CMake 3.20.2, installed with Homebrew
*   yaml-cpp 0.6.3, compiled from source
*   Trilinos 12.18.1, compiled from source
*   Boost 1.76.0, compiled from source
*   BLAS++ 2021.04.00, compiled from source
*   LAPACK++ 2021.04.00, compiled from source

For example configuration scripts to install these dependencies, see [this
repository](https://github.com/saibalde/softwares). Adjust the configuration
scripts as necessary for your build environment.

## Building

Once the dependencies are installed, build the program as:

```sh
mkdir build # out of source build
cd build
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER=/path/to/cxx/compiler \
    -D CNQS_BUILD_PDESOLVER=ON \
    -D CNQS_BUILD_VMCSOLVER=ON \
    -D CNQS_BUILD_DOCS=ON \
    -D yaml-cpp_ROOT=/path/to/yaml-cpp/install/prefix \
    -D Trilinos_ROOT=/path/to/trilinos/install/prefix \
    -D Boost_ROOT=/path/to/boost/install/prefix \
    -D blaspp_ROOT=/path/to/blaspp/install/prefix \
    -D lapackpp_ROOT=/path/to/lapackpp/install/prefix \
    ..
make
```

Note that this will build both the PDE and VMC eigensolvers. You can also build
a single type of eigensolvers by passing the appropriate CMake command line
argument:

*   Pass `CNQS_BUILD_PDESOLVER=OFF` to switch off building the PDE eigensolver.
    You do not need to set `Trilinos_ROOT` in this case.

*   Pass `CNQS_BUILD_VMCSOLVER=OFF` to switch off building the VMC eigensolver.
    You do not need to set `Boost_ROOT`, `blaspp_ROOT` or `lapackpp_ROOT` in
    this case.

The following executables are created in the build directory:

*   If `CNQS_BUILD_PDESOLVER=ON` then

    *   `./app/pdesolver_fd` implements PDE eigensolver using finite difference
        scheme to discretize the quantum state and the Hamiltonian.

    *   `./app/pdesolver_sd` implements PDE eigensolver using spectral
        difference scheme (using truncated Fourier series) to discretize the
        quantum state and the Hamiltonian.

*   If `CNQS_BUILD_VMCSOLVER=ON` then

    *   `./app/vmcsolver` implements VMC eigensolver.

## Running

Two example configurations `hamiltonian.yaml` and `config.yaml` are supplied in
the repository root directory. These are copied over in the build directory.

### PDE Eigensolver with Finite Difference Discretization

Run the executable from inside the build directory as:

```sh
./app/pdesolver_fd \
    --hamiltonian-file-name=hamiltonian.yaml \
    --num-grid-point=128 \
    --max-power-iter=10000 \
    --tol-power-iter=1.0e-15 \
    --max-cg-iter=10000 \
    --tol-cg-iter=1.0e-15 \
    --ground-state-file-name=ground_state.mm
```

The `--ground-state-file-name` can be ommitted; in this the ground state is not
saved.

### PDE Eigensolver with Spectral Difference Discretization

Run the executable from inside the build directory as:

```sh
./app/pdesolver_fd \
    --hamiltonian-file-name=hamiltonian.yaml \
    --max-frequency=32 \
    --max-power-iter=10000 \
    --tol-power-iter=1.0e-15 \
    --max-cg-iter=10000 \
    --tol-cg-iter=1.0e-15 \
    --ground-state-file-name=ground_state.mm
```

The `--ground-state-file-name` can be ommitted; in this the ground state is not
saved.

### VMC Eigensolver

Run the executable from inside the build directory as:

```sh
./app/vmcsolver config.yaml
```

This will create `<timestamp>_config.yaml` storing the run configuration (which
can be used for later re-runs) and `<timestamp>_output.csv` storing the outputs
from the current run.

Note that the Hamiltonian can be specified directly in `config.yaml`. See the
`<timestamp>_config.yaml` generated above for an example.

Finally, running
```sh
./app/vmcsolver <config.yaml> <output_prefix>
```
will create `<output_prefix>_config.yaml` and `<output_prefix>_output.csv`
instead of the timestamped files.
