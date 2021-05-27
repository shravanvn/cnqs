import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml


def main(config, max_lag):
    data_dir = config['output']['prefix']
    sgd_step_list = np.arange(start=0,
                              stop=config['gradient_descent']['num_steps'] + 1,
                              step=config['output']['frequency'])

    os.makedirs('{}/plots/'.format(data_dir), exist_ok=True)

    for sgd_step in sgd_step_list:
        samples = np.loadtxt('{}/samples/step_{}.txt'.format(data_dir,
                                                             sgd_step))
        m, n = samples.shape

        # trace plot
        for j in range(n):
            fig, ax = plt.subplots()
            ax.plot(np.arange(m), samples[:, j])
            ax.set_xlabel('MCMC Step')
            ax.set_ylabel('State[{}]'.format(j))
            fig.savefig('{}/plots/step_{}_trace_{}.pdf'.format(data_dir,
                                                               sgd_step,
                                                               j))
            plt.close(fig)

        # autocorrelation
        mu = samples.mean(axis=0)[None, :]

        autocorr = np.zeros((max_lag + 1, n), dtype=np.double)
        for k in range(max_lag + 1):
            numerator = (samples[k:, :] - mu) * (samples[:m-k, :] - mu)
            denominator = (samples - mu)**2
            autocorr[k, :] = numerator.mean(axis=0) / denominator.mean(axis=0)

        fig, ax = plt.subplots()
        ax.plot(np.arange(max_lag + 1), autocorr)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        fig.savefig('{}/plots/step_{}_autocorr.pdf'.format(data_dir, sgd_step))
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_file',
        type=str,
        help='path to config file generated by vmcsolver'
    )
    parser.add_argument(
        '--max_lag',
        type=int,
        help='maximum lag used in autocorrelation computation'
    )

    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    main(config, args.max_lag)
