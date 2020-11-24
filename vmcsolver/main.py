import argparse
import os

from numpy.random import seed
from sampler import metropolis_sampler
from logger import Logger
from util import read_config
from nqs import NQS, propose_update
from wavefunction import local_energy, log_psi, log_psi_vars

from optimization import stoch_reconfig

seed(1)


def main():
    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config.yaml')
    args = parser.parse_args()
    config = read_config(args.config_path)

    # logging
    logger = Logger(config['logdir'])
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
    main()
