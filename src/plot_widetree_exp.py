import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 15})

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def plot_data(seeds):
    mle_weights = []
    moment_based_weights = []

    for seed in seeds:
        mle_weights.append(np.load(f"data/mle_{seed}.npy"))
        moment_based_weights.append(np.load(f"data/moment_based_{seed}.npy"))

    mle_mean_weights, mle_std_weights = np.mean(mle_weights, axis=0), np.std(mle_weights, axis=0)
    moment_based_mean_weights, moment_based_std_weights = np.mean(moment_based_weights, axis=0), np.std(moment_based_weights, axis=0)

    plt.clf()
    xrange = np.arange(mle_mean_weights.shape[0])
    
    for model_idx in range(mle_mean_weights.shape[1]):
        color = CB_color_cycle[model_idx]
        plt.plot(xrange, mle_mean_weights[:, model_idx], color=color, label=f"Good Model" if model_idx > 0 else f"Bad Model")
        plt.fill_between(xrange, mle_mean_weights[:, model_idx] - mle_std_weights[:, model_idx], mle_mean_weights[:, model_idx] + mle_std_weights[:, model_idx], color=color, alpha=0.2)

    plt.xlabel("Number of Iterations")
    plt.ylabel("Probability using classification loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("mle_widetree_exp.png")

    plt.clf()    
    for model_idx in range(moment_based_mean_weights.shape[1]):
        color = CB_color_cycle[model_idx]
        plt.plot(xrange, moment_based_mean_weights[:, model_idx], color=color, label=f"Good Model" if model_idx > 0 else f"Bad Model")
        plt.fill_between(xrange, moment_based_mean_weights[:, model_idx] - moment_based_std_weights[:, model_idx], moment_based_mean_weights[:, model_idx] + moment_based_std_weights[:, model_idx], color=color, alpha=0.2)

    plt.xlabel("Number of Iterations")
    plt.ylabel("Probability using moment based loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("moment_based_widetree_exp.png")


if __name__ == "__main__":
    seeds = range(10)
    plot_data(seeds)
    