import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def main(data_dir, sgd_step, max_lag):
    os.makedirs('{}/plots/'.format(data_dir), exist_ok=True)

    samples = np.loadtxt('{}/samples/step_{}.dat'.format(data_dir, sgd_step))
    m, n = samples.shape

    # trace plot
    for j in range(n):
        fig, ax = plt.subplots()
        ax.plot(np.arange(m), samples[:, j])
        ax.set_xlabel('MCMC Step')
        ax.set_ylabel('State[{}]'.format(j))
        fig.savefig('{}/plots/step_{}_trace_{}.pdf'.format(data_dir, sgd_step,
                                                           j))
        plt.close(fig)

    # autocorrelation
    mu = samples.mean(axis=0)[None, :]

    autocorr = np.zeros((max_lag + 1, n), dtype=np.double)
    for k in range(max_lag + 1):
        autocorr[k, :] = \
            ((samples[k:, :] - mu) * (samples[:m-k, :] - mu)).mean(axis=0) / \
            ((samples - mu)**2).mean(axis=0)

    fig, ax = plt.subplots()
    ax.plot(np.arange(max_lag + 1), autocorr)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    fig.savefig('{}/plots/step_{}_autocorr.pdf'.format(data_dir, sgd_step))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        type=str,
        help='path to directory containing sample files'
    )
    parser.add_argument(
        '--sgd_step',
        type=int,
        help='SGD step whose sample autocorrelation will be computed'
    )
    parser.add_argument(
        '--max_lag',
        type=int,
        help='Maximum lag in autocorrelation computation'
    )

    args = parser.parse_args()

    main(args.data_dir, args.sgd_step, args.max_lag)
