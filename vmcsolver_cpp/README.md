# VMC Eigensolver

## Dependencies

To build this program, the following are required:

*   C++11 compatible compiler
*   [Boost](https://www.boost.org/) (only the math_tr1 sublibrary is sufficient)
*   [BLAS++](https://bitbucket.org/icl/blaspp/)
*   [LAPACK++](https://bitbucket.org/icl/lapackpp/)
*   [yaml-cpp](https://github.com/jbeder/yaml-cpp)
*   [CMake](http://cmake.org/) >= 3.12.0

The code is tested on macOS Big Sur with

*   AppleClang 12.0.5
*   Boost 1.75.0
*   BLAS++ 2020.10.02
*   LAPACK++ 2020.10.02
*   yaml-cpp 0.6.3
*   CMake 3.20.2

## Building

Once the dependencies are installed, build the program as

```sh
mkdir build # out of source build
cd build
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER=/path/to/cxx/compiler \
    -D Boost_ROOT=/path/to/boost/install/prefix \
    -D blaspp_ROOT=/path/to/blaspp/install/prefix \
    -D lapackpp_ROOT=/path/to/lapackpp/install/prefix \
    -D yaml-cpp_ROOT=/path/to/yaml-cpp/install/prefix \
    ..
make
```

This will create the `app/vmcsolver` executable in the build directory.

## Running

The build directory will also contain two configuration files: `config.yaml`
specifying the VMC parameters, and `hamiltonian.yaml` specifying the rotor
Hamiltonian. To run the simulation, execute
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
