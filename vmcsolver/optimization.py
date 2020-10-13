import numpy as np


def gradients(step, averages, logger=None):
    O_av, OO_av, EO_av, E_av = averages
    grads = EO_av - O_av * E_av
    fisher = OO_av - np.outer(O_av, O_av)
    if logger:
        print('average energy on step {}: {}'.format(step, E_av))
        logger.log_scalar('E_av', E_av, step)
        logger.log_scalar('grads_norm', np.linalg.norm(grads), step)
    return grads, fisher


def stoch_reconfig(step, nqs, averages, config, logger=None):
    grads, fisher = gradients(step, averages, logger)
    fisher_reg = fisher + config['sr_reg'] * np.eye(len(nqs.vars))
    vars_new = nqs.vars - config['lr'] * np.linalg.solve(fisher_reg, grads)
    return vars_new
