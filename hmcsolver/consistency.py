import argparse
import numpy as np

from copy import deepcopy
from numpy.random import seed
from numpy.random import uniform
from nqs import NQS
from wavefunction import local_energy, log_psi, amplitude, log_psi_vars, local_kinetic_energy
from util import unroll, angle, read_config

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default='./config.yaml')
args = parser.parse_args()
config = read_config(args.config_path)


seed(1)
epsilon = 1e-4  # for finite difference

U, J = config['U'], config['J']
n, H = config['num_visible'], config['num_hidden']
state = unroll(uniform(-np.pi, np.pi, size=n))
cs = uniform(-1, 1, size=(n, 2))
bs = uniform(-1, 1, size=(H, 2))
weights = uniform(-1, 1, size=(H, n))
var_list = [cs, bs, weights]
vars = np.concatenate([np.reshape(var, -1) for var in var_list])
nqs = NQS(config, state=state, vars=vars)


def local_energy_test():

    kinetic = 0.0
    for site in range(len(state)):
        state_plus, state_minus = np.array(state), np.array(state)
        nqs_plus, nqs_minus = deepcopy(nqs), deepcopy(nqs)
        nqs_plus.update_state(new_val=state[site] + epsilon, site=site)
        nqs_minus.update_state(new_val=state[site] - epsilon, site=site)

        # psi = np.exp(log_psi(nqs))
        # psi_plus = np.exp(log_psi(nqs_plus))
        # psi_minus = np.exp(log_psi(nqs_minus))
        psi = amplitude(nqs)
        psi_plus = amplitude(nqs_plus)
        psi_minus = amplitude(nqs_minus)
        theta = state[site]

        # HMC adjustments
        # adj_1 = (2 / np.square(np.pi)) * np.power(np.cosh(theta), 3) * np.sinh(theta)
        # adj_2 = np.power(np.cosh(theta), 4) / np.square(np.pi)
        # kinetic -= adj_2 * (psi_plus + psi_minus - 2 * psi) / (psi * np.square(epsilon))
        # kinetic -= adj_1 * (psi_plus - psi_minus) / (psi * 2 * epsilon)

        kinetic -= (psi_plus + psi_minus - 2 * psi) / (psi * np.square(epsilon))


    potential = 0.0
    for site in range(len(state) - 1):
        potential -= np.cos(angle(state[site]) - angle(state[site + 1]))

    local_energy_fd = 2 * J * potential + (U / 2) * kinetic

    print('local_energy', local_energy(nqs, config))
    print('local_energy_fd', local_energy_fd)
    print('local_kinetic_energy', -local_kinetic_energy(nqs))
    print('local_kinetic_energy_fd', kinetic)


def log_psi_vars_test():
    grad_fd = []
    for i in range(len(vars)):
        vars_plus = deepcopy(vars)
        vars_minus = deepcopy(vars)
        vars_plus[i] += epsilon
        vars_minus[i] -= epsilon
        nqs_plus = NQS(config, state=state, vars=vars_plus)
        nqs_minus = NQS(config, state=state, vars=vars_minus)
        psi = amplitude(nqs)
        psi_plus = amplitude(nqs_plus)
        psi_minus = amplitude(nqs_minus)
        deriv = (psi_plus - psi_minus) / (psi * 2 * epsilon)
        grad_fd.append(deriv)

    grad = log_psi_vars(nqs)
    grad_fd = np.array(grad_fd)
    delta = grad - grad_fd
    print('maximum absolute deviation', np.max(np.abs(delta)))

local_energy_test()
log_psi_vars_test()

# local_energy -1.8275479646890944
# local_energy_fd -1.8275479775172871
# maximum absolute deviation 1.5811842946078514e-09
