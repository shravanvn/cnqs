import numpy as np

from util import average, standard_deviation


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
    history = []
    operators = sampling_functions(local_energy, log_psi_vars, config)

    start = config['metropolis']['warm_steps']
    stop = config['metropolis']['num_steps']
    step = config['metropolis']['cherry_pick']

    last_nqs = nqs_init
    for t in range(stop):
        new_nqs = propose_update(last_nqs, config)
        if accept(new_nqs, last_nqs, log_psi):
            last_nqs = new_nqs
            num_acceptances += 1

        if t >= start and (t - start) % step == 0:
            history.append(last_nqs)

    if logger:
        logger.log_scalar('energy_std', std(operators[-1], history))
        logger.log_scalar('acceptance_rate', num_acceptances / stop)

    return [mean(op, history) for op in operators], last_nqs


def accept(new_nqs, last_nqs, log_psi):
    log_psi_new = log_psi(new_nqs)
    log_psi_old = log_psi(last_nqs)
    log_p = np.log(np.random.random())
    return log_psi_new > log_psi_old + 0.5 * log_p


def mean(op, history):
    return average([op(nqs) for nqs in history])


def std(op, history):
    return standard_deviation([op(nqs) for nqs in history])


def sampling_functions(local_energy, log_psi_vars, config):

    def energy(nqs):
        return local_energy(nqs, config)

    def log_psi_vars_tensor(nqs):
        grad = log_psi_vars(nqs)
        return np.outer(grad, grad)

    def energy_log_psi_vars(nqs):
        return energy(nqs) * log_psi_vars(nqs)

    return [log_psi_vars, log_psi_vars_tensor, energy_log_psi_vars, energy]
