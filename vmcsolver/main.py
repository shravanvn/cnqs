import argparse
import numpy as np
import time
import datetime
import yaml

from logger import Logger
from nqs import NQS, propose_update
from sampler import metropolis_sampler
from wavefunction import local_energy, log_psi, log_psi_vars
from optimization import stoch_reconfig

np.random.seed(1)


def main(config):
    # logging
    logger = Logger(config['output_prefix'] + '_output.csv')
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
    for step in range(1, config['gradient_descent']['num_steps'] + 1):
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

        if step % config['logging']['stdout_frequency'] == 0:
            print('{:13d}'.format(step), end='', flush=True)
            logger.write_step_to_stdout(stdout_vars)

    toc = time.time()

    print('-------------------------------------------------------', flush=True)
    print('elapsed_time = {:f} sec'.format(toc - tic), flush=True)
    print('=======================================================', flush=True)


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        default='config.yaml',
        help='Path to YAML config',
        type=str
    )
    parser.add_argument(
        '--hamiltonian',
        default=None,
        help="Path to YAML description of the Hamiltonian",
        type=str
    )
    parser.add_argument(
        '--num_hidden',
        default=None,
        help='Number of hidden nodes',
        type=int
    )
    parser.add_argument(
        '--num_steps',
        default=None,
        help='Number of gradient descent steps',
        type=int
    )
    parser.add_argument(
        '--lr',
        default=None,
        help='Learning rate in gradient descent',
        type=float
    )
    parser.add_argument(
        '--sr_reg',
        default=None,
        help='Stochastic reconfiguration regularization parameter',
        type=float
    )
    parser.add_argument(
        '--metropolis_steps',
        default=None,
        help='Number of Metropolis samples to generate',
        type=int
    )
    parser.add_argument(
        '--warm_steps',
        default=None,
        help='Number of initial Metropolis samples to discard',
        type=int
    )
    parser.add_argument(
        '--cherry_pick',
        default=None,
        help='Frequency of cherry-picking Metropolis samples',
        type=int
    )
    parser.add_argument(
        '--bump_size',
        default=None,
        help='Bump parameter in Metropolis proposal',
        type=float
    )
    parser.add_argument(
        '--logger_stdout_frequency',
        default=None,
        help='Frequency of printing to standard output stream',
        type=int
    )
    parser.add_argument(
        '--output_prefix',
        default='runs/' + datetime.datetime.now().isoformat(),
        help='Prefix to add before names of output files',
        type=str
    )

    args = vars(parser.parse_args())

    # construct config
    config = {}
    with open(args['config_file'], mode='r') as file:
        config.update(yaml.safe_load(file))

    if args['hamiltonian'] is not None:
        with open(args['hamiltonian'], mode='r') as file:
            config['hamiltonian'] = yaml.safe_load(file)
    elif type(config['hamiltonian']) == str:
        with open(config['hamiltonian'], mode='r') as file:
            config['hamiltonian'] = yaml.safe_load(file)

    if args['num_hidden'] is not None:
        config['rbm']['num_hidden'] = args['num_hidden']

    if args['num_steps'] is not None:
        config['gradient_descent']['num_steps'] = args['num_steps']
    if args['lr'] is not None:
        config['gradient_descent']['lr'] = args['lr']

    if args['sr_reg'] is not None:
        config['stoch_reconfig']['sr_reg'] = args['sr_reg']

    if args['metropolis_steps'] is not None:
        config['metropolis']['num_steps'] = args['metropolis_steps']
    if args['warm_steps'] is not None:
        config['metropolis']['warm_steps'] = args['warm_steps']
    if args['cherry_pick'] is not None:
        config['metropolis']['cherry_pick'] = args['cherry_pick']
    if args['bump_size'] is not None:
        config['metropolis']['bump_size'] = args['bump_size']

    if args['logger_stdout_frequency'] is not None:
        config['logging']['stdout_frequency'] = args['logger_stdout_frequency']

    config['output_prefix'] = args['output_prefix']

    # write config to file
    with open(config['output_prefix'] + '_config.yaml', mode='w') as file:
        yaml.dump(config, file)

    # execute main program
    main(config)
