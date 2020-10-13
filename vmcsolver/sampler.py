import numpy as np

from util import average


def metropolis_sampler(step,
                       local_energy,
                       log_psi,
                       log_psi_vars,
                       nqs_init,
                       config,
                       propose_update,
                       logger=None):
    """
    :param step: gradient iteration
    :param log_psi: function object
    :param log_psi_vars: function object
    :param local_energy: function object
    :param nqs_init: initial state of Markov chain
    :param config: config data
    :return: list of expectation values of target functions
    """

    num_acceptances = 0
    history = [nqs_init]
    operators = sampling_functions(local_energy, log_psi_vars, config)

    for t in range(config['metropolis_steps']):
        last_nqs = history[-1]
        new_nqs = propose_update(last_nqs, config)
        if accept(new_nqs, last_nqs, log_psi):
            history.append(new_nqs)
            num_acceptances += 1
        else:
            history.append(last_nqs)

    if logger:
        logger.log_scalar('acceptance_rate', num_acceptances / config['metropolis_steps'], step)
        print('acceptance_rate', num_acceptances / config['metropolis_steps'])
    start = config['warm_steps']
    return [mean(op, history[start:]) for op in operators], history[-1]


def accept(new_nqs, last_nqs, log_psi):
    log_psi_new = log_psi(new_nqs)
    log_psi_old = log_psi(last_nqs)
    log_p = np.log(np.random.random())
    return log_psi_new > log_psi_old + 0.5 * log_p


def mean(op, history):
    return average([op(nqs) for nqs in history])


def sampling_functions(local_energy, log_psi_vars, config):

    def energy(nqs):
        return local_energy(nqs, config)

    def log_psi_vars_tensor(nqs):
        grad = log_psi_vars(nqs)
        return np.outer(grad, grad)

    def energy_log_psi_vars(nqs):
        return energy(nqs) * log_psi_vars(nqs)

    return [log_psi_vars, log_psi_vars_tensor, energy_log_psi_vars, energy]
