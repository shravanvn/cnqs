import numpy as np
import yaml

from scipy.special import iv


def average(x):
    return sum(x) / len(x)


def g_func(x):
    return iv(1, x) / iv(0, x)


def cartesian(state):
    angles = angle(state)
    cosines = np.cos(angles)
    sines = np.sin(angles)
    return np.stack([cosines, sines], axis=1)


def perp(state):
    state_perp = np.copy(state)
    state_perp[:, 0], state_perp[:, 1] = -state[:, 1], state[:, 0]
    return state_perp


def center(angle):
    return ((angle + np.pi) % (2*np.pi)) - np.pi


# HMC helper
def unroll(angle):
    # return np.arctanh(angle / np.pi)
    return angle


# HMC helper
def angle(real):
    # return np.pi * np.tanh(real)
    return real


def sech(angle):
    return 1.0 / np.cosh(angle)


def read_config(config_path):
    with open(config_path, "r") as f:
        config_data = f.read()
        params = yaml.load(config_data, Loader=yaml.CLoader)
    return params


