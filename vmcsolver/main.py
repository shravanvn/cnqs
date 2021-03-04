import argparse
import numpy as np

from util import read_config
from logger import Logger
from nqs import NQS, propose_update
from sampler import metropolis_sampler
from wavefunction import local_energy, log_psi, log_psi_vars
from optimization import stoch_reconfig

np.random.seed(1)


def main(run_dir):
    # read configuration
    config = read_config(run_dir + "/config.yaml")

    # logging
    logger = Logger(run_dir)
    logger.set_variables(['acceptance_rate', 'b_norm', 'c_norm', 'energy_avg', 'energy_std', 'grad_norm'])
    logger.write_header()

    nqs = NQS(config=config)

    for step in range(1, config['num_step'] + 1):
        print('step = {:d}'.format(step))
        averages, nqs = metropolis_sampler(step,
                                           local_energy,
                                           log_psi,
                                           log_psi_vars,
                                           nqs_init=nqs,
                                           config=config,
                                           propose_update=propose_update,
                                           logger=logger)
        vars = stoch_reconfig(step, nqs, averages, config, logger)
        nqs = NQS(config=config, state=nqs.state, vars=vars)

        logger.write_step(step)


if __name__ == "__main__":
    # parse run directory from command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', default='./')
    args = parser.parse_args()

    main(args.run_dir)
