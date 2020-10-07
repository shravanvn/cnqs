# HMCSolver

## Installation

With the `conda` package manager (e.g. using Anaconda distribution), the
following packages need to be installed
-   numpy
-   scipy
-   tensorflow
-   pyyaml
Then to execute the code, run
```sh
python main.py --config_path config.yaml
```
To visualize the diagnostics using tensorboard, run
```sh
tensorboard --logdir tensorboard/
```
and go to the URL in the output.
