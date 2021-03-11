# VMCSolver

## Environment Setup

Assuming you are using the `conda` package manager (from Anaconda distribution,
or its minimal version miniconda3), execute
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
python main.py --run_dir /path/to/run/dir
```
where `/path/to/run/dir` contains the `config.yaml` file; This will generate an
`output.csv` file in the same run directory that will record the average
energy, energy standard deviation, gradient norms etc. per gradient descent
step.

Then, to plot the outputs, run
```sh
python plot.py --run_dir /path/to/run/dir
```
This will read the `output.csv` file in `/path/to/run/dir` and create
timeseries plots for the output variables. The figure will be saved as
`output.pdf` in the run directory.

## Profiling

To profile the run, `cProfile` module can be used; inside the `cnqs` environment, run
```sh
python -m cProfile -o /path/to/run/dir/cprofile.bin main.py --run_dir /path/to/run/dir
```

This profile can be visualized by
```sh
snakeviz /path/to/run/dir/cprofile.bin
```
