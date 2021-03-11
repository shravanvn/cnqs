import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main(run_dir):
    # read output
    df = pd.read_csv(run_dir + "/output.csv")

    # generate plot
    fig, ax = plt.subplots(2, 3, figsize=(9.6, 4.8))

    ax[0, 0].plot(df["step"], df["energy_avg"])
    ax[0, 0].set_xlabel("Gradient Descent Step")
    ax[0, 0].set_ylabel("Average Energy")

    ax[0, 1].semilogy(df["step"], df["energy_std"])
    ax[0, 1].set_xlabel("Gradient Descent Step")
    ax[0, 1].set_ylabel("Energy Standard Deviation")

    ax[0, 2].semilogy(df["step"], df["grad_norm"])
    ax[0, 2].set_xlabel("Gradient Descent Step")
    ax[0, 2].set_ylabel("Gradient Norm")

    ax[1, 0].plot(df["step"], df["acceptance_rate"])
    ax[1, 0].set_xlabel("Gradient Descent Step")
    ax[1, 0].set_ylabel("MCMC Acceptance Rate")

    ax[1, 1].plot(df["step"], df["c_norm"])
    ax[1, 1].set_xlabel("Gradient Descent Step")
    ax[1, 1].set_ylabel("Visible Bias Norm")

    ax[1, 2].plot(df["step"], df["b_norm"])
    ax[1, 2].set_xlabel("Gradient Descent Step")
    ax[1, 2].set_ylabel("Hidden Bias Norm")

    plt.tight_layout()

    # save plot
    fig.savefig(run_dir + "/output.pdf")


if __name__ == "__main__":
    # parse run directory from command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', default='./')
    args = parser.parse_args()

    main(args.run_dir)
