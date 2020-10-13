import argparse
import os

import time
from datetime import datetime
from numpy.random import seed
from sampler import metropolis_sampler
from logger import Logger
from util import read_config

from optimization import stoch_reconfig

seed(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config.yaml')
    args = parser.parse_args()
    config = read_config(args.config_path)
    now = datetime.now().replace(microsecond=0).isoformat()
    tbdir = os.path.join(config['tensorboard_path'], now)
    logger = Logger(tbdir)

    if config['debug']:
        from nqs_sho import NQS, propose_update
        from wavefunction_sho import local_energy, log_psi, log_psi_vars
    else:
        from nqs import NQS, propose_update
        from wavefunction import local_energy, log_psi, log_psi_vars

    nqs = NQS(config=config)

    start = time.time()
    for step in range(1, 1000):
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
    end = time.time()
    print('Calculation time', end - start)


if __name__ == "__main__":
    main()
