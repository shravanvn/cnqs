import numpy as np


def gradients(step, averages, logger=None):
    O_av, OO_av, EO_av, E_av = averages
    grads = EO_av - O_av * E_av
    fisher = OO_av - np.outer(O_av, O_av)

    if logger:
        logger.log_scalar('energy_avg', E_av)
        logger.log_scalar('grad_norm', np.linalg.norm(grads))

    return grads, fisher


def stoch_reconfig(step, nqs, averages, config, logger=None):
    grads, fisher = gradients(step, averages, logger)
    fisher_reg = fisher + \
        config['stoch_reconfig']['sr_reg'] * np.eye(len(nqs.vars))
    vars_new = nqs.vars - \
        config['gradient_descent']['lr'] * np.linalg.solve(fisher_reg, grads)

    if logger:
        logger.log_scalar('b_norm', np.sqrt(np.sum(nqs.bs**2)))
        logger.log_scalar('c_norm', np.sqrt(np.sum(nqs.cs**2)))

    return vars_new
