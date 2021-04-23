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
    logger.set_variables(['acceptance_rate', 'b_norm',
                         'c_norm', 'energy_avg', 'energy_std', 'grad_norm'])
    logger.write_header()

    stdout_vars = ['energy_avg', 'energy_std', 'grad_norm']
    print('=======================================================', flush=True)
    print('{:>13s}'.format('step'), end='', flush=True)
    logger.write_header_to_stdout(stdout_vars)
    print('-------------------------------------------------------', flush=True)

    # initialize neural quantum state
    nqs = NQS(config=config)

    # main loop
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
            print('{:13d}'.format(step), end='', flush=True)
            logger.write_step_to_stdout(stdout_vars)

    toc = time.time()

    print('-------------------------------------------------------', flush=True)
    print('elapsed_time = {:f} sec'.format(toc - tic), flush=True)
    print('=======================================================', flush=True)


if __name__ == "__main__":
    # parse run directory from command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config.yaml')
    parser.add_argument('--output_file', default='output.csv')
    args = parser.parse_args()

    main(args.config_file, args.output_file)
