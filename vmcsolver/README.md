# VMCSolver

## Dependencies

This VMC code depends on the following python packages:

*   Core simulation:
    *   numpy
    *   scipy
    *   pyyaml

*   Plotting simulation outputs:
    *   pandas
    *   matplotlib

*   Profile visualization:
    *   snakeviz

A basic method for setting up this environment is to use the [conda package
manager](https://docs.conda.io/en/latest/) and run
```sh
conda create -n cnqs python numpy scipy pyyaml matplotlib pandas snakeviz
```
This will create the `cnqs` environment which can be activated by
```sh
conda activate cnqs
```

## Running the Simulations

To run variational Monte-Carlo simulation, execute inside the `cnqs` environment
```sh
python main.py [--config_file /path/to/config.yaml] [--output_file /path/to/output.csv]
```
where `config.yaml` specifies simulation parameters and various quantities of
interest (e.g. average energy, gradient norm) are written to the `output.csv`
file.

To visualize the simulation outputs, run
```sh
python plot.py [--output_file /path/to/output.csv] [--figure_name /path/to/output.pdf]
```
This will create an `output.pdf` containing timeseries plots of various
quantities of interest. For convenience, if the `--figure_name
/path/to/output.pdf` part is ommitted, then the generated figure will share the
same base name as the output file.

## Profiling

To profile the simulation, `cProfile` module can be used; inside the `cnqs`
environment, run
```sh
python -m cProfile -o /path/to/cprofile.bin main.py [--config_file /path/to/config.yaml] [--output_file /path/to/output.csv]
```
The generated profile can be visualized by
```sh
snakeviz /path/to/cprofile.bin
```
