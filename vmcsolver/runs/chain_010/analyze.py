import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(run_id_list, run_label_list, figure_name_prefix, rolling_window):
    num_run = len(run_id_list)
    assert num_run == len(run_label_list)

    energy_avg_fig,   energy_avg_ax   = plt.subplots()
    energy_std_fig,   energy_std_ax   = plt.subplots()
    grad_norm_fig,    grad_norm_ax    = plt.subplots()
    acceptance_fig,   acceptance_ax   = plt.subplots()
    visible_bias_fig, visible_bias_ax = plt.subplots()
    hidden_bias_fig,  hidden_bias_ax  = plt.subplots()

    for n in range(num_run):
        df = pd.read_csv('run_{:s}/output.csv'.format(run_id_list[n]))

        energy_avg_ax.plot(    df['step'], df['energy_avg'].rolling(rolling_window).mean(),      label=run_label_list[n])
        energy_std_ax.semilogy(df['step'], df['energy_std'].rolling(rolling_window).mean(),      label=run_label_list[n])
        grad_norm_ax.semilogy( df['step'], df['grad_norm'].rolling(rolling_window).mean(),       label=run_label_list[n])
        acceptance_ax.plot(    df['step'], df['acceptance_rate'].rolling(rolling_window).mean(), label=run_label_list[n])
        visible_bias_ax.plot(  df['step'], df['c_norm'].rolling(rolling_window).mean(),          label=run_label_list[n])
        hidden_bias_ax.plot(   df['step'], df['b_norm'].rolling(rolling_window).mean(),          label=run_label_list[n])

    energy_avg_ax.legend(  loc='best')
    energy_std_ax.legend(  loc='best')
    grad_norm_ax.legend(   loc='best')
    acceptance_ax.legend(  loc='best')
    visible_bias_ax.legend(loc='best')
    hidden_bias_ax.legend( loc='best')

    energy_avg_ax.set_xlabel(  'Gradient Descent Step')
    energy_std_ax.set_xlabel(  'Gradient Descent Step')
    grad_norm_ax.set_xlabel(   'Gradient Descent Step')
    acceptance_ax.set_xlabel(  'Gradient Descent Step')
    visible_bias_ax.set_xlabel('Gradient Descent Step')
    hidden_bias_ax.set_xlabel( 'Gradient Descent Step')

    energy_avg_ax.set_ylabel(  'Average Energy')
    energy_std_ax.set_ylabel(  'Energy Standard Deviation')
    grad_norm_ax.set_ylabel(   'Gradient Norm')
    acceptance_ax.set_ylabel(  'Acceptance Rate')
    visible_bias_ax.set_ylabel('Visible Bias Norm')
    hidden_bias_ax.set_ylabel( 'Hidden Bias Norm')

    energy_avg_fig.savefig(  '{:s}_average_energy.pdf'.format(figure_name_prefix))
    energy_std_fig.savefig(  '{:s}_energy_standard_deviation.pdf'.format(figure_name_prefix))
    grad_norm_fig.savefig(   '{:s}_gradient_norm.pdf'.format(figure_name_prefix))
    acceptance_fig.savefig(  '{:s}_acceptance_rate.pdf'.format(figure_name_prefix))
    visible_bias_fig.savefig('{:s}_visible_bias_norm.pdf'.format(figure_name_prefix))
    hidden_bias_fig.savefig( '{:s}_hidden_bias_norm.pdf'.format(figure_name_prefix))

    plt.close(energy_avg_fig)
    plt.close(energy_std_fig)
    plt.close(grad_norm_fig)
    plt.close(acceptance_fig)
    plt.close(visible_bias_fig)
    plt.close(hidden_bias_fig)

if __name__ == '__main__':
    # hidden-to-visible node density
    main(['01', '02', '03', '04', '05'], ['density =  2', 'density =  4', 'density =  6', 'density =  8', 'density = 10'], 'density_1', 100)
    main(['06', '07', '08', '09', '10'], ['density =  2', 'density =  4', 'density =  6', 'density =  8', 'density = 10'], 'density_2', 100)

    # learning rate
    main(['01', '06'], ['learning_rate = 1.0e-03', 'learning_rate = 2.5e-03'], 'learning_rate_1', 100)
    main(['02', '07'], ['learning_rate = 1.0e-03', 'learning_rate = 2.5e-03'], 'learning_rate_2', 100)
    main(['03', '08'], ['learning_rate = 1.0e-03', 'learning_rate = 2.5e-03'], 'learning_rate_3', 100)
    main(['04', '09'], ['learning_rate = 1.0e-03', 'learning_rate = 2.5e-03'], 'learning_rate_4', 100)
    main(['05', '10'], ['learning_rate = 1.0e-03', 'learning_rate = 2.5e-03'], 'learning_rate_5', 100)
