import argparse
import numpy as np
import time

from util import read_config
from logger import Logger
from nqs import NQS, propose_update
from sampler import metropolis_sampler
from wavefunction import local_energy, log_psi, log_psi_vars
from optimization import stoch_reconfig

np.random.seed(1)


def main(config_file, output_file):
    # read configuration
    config = read_config(config_file)

    # logging
    logger = Logger(output_file)
    logger.set_variables(['acceptance_rate', 'b_norm', 'c_norm', 'energy_avg', 'energy_std', 'grad_norm'])
    logger.write_header()

    nqs = NQS(config=config)

    tic = time.time()
    for step in range(1, config['num_step'] + 1):
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

        if step % config['logger_stdout_frequency'] == 0:
            print("step = {:d}, ".format(step), end="")
            logger.write_stdout(['energy_avg', 'energy_std', 'grad_norm'])

    toc = time.time()

    print("================================================================================")
    print("Finished!")
    print("--------------------------------------------------------------------------------")
    print("elapsed_time = {:f} sec, ".format(toc - tic), end="")
    logger.write_stdout(['energy_avg', 'energy_std', 'grad_norm'])
    print("================================================================================")


if __name__ == "__main__":
    # parse run directory from command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config.yaml')
    parser.add_argument('--output_file', default='output.csv')
    args = parser.parse_args()

    main(args.config_file, args.output_file)
