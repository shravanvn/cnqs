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
python main.py --config_file /path/to/config.yaml --output_prefix /path/to/output/<run_id>
```
This will execute the VMC simulation as specified in the specified
`config.yaml` and save the outputs (average energy, energy standard deviation
etc.) in `/path/to/output/<run_id>_output.csv`.  This will also create
`/path/to/output/<run_id>_config.yaml` which can be used to rerun the
simulation with exactly the same parameters.

*   If no `--config-file` is specified, the `config.yaml` file from this
    directory is used

*   Parameters specified inside the configuration file can be overriden via
    command line arguments. Run `python main.py --help` for a full listing of
    available options.

*   If no `--output_prefix` is specified, the prefix defaults to
    `runs/<timestamp>`

To visualize the simulation outputs, run
```sh
python plot.py --output_prefix /path/to/output/<run_id>
```
This will create an `/path/to/output/<run_id>_output.pdf` containing timeseries
plots of various quantities of interest stored in
`/path/to/output/<run_id>_output.csv`.

## Profiling

To profile the simulation, `cProfile` module can be used; inside the `cnqs`
environment, run
```sh
python -m cProfile -o /path/to/cprofile.bin main.py [options]
```
The generated profile can be visualized by
```sh
snakeviz /path/to/cprofile.bin
```
