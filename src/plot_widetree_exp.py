import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 15})

def plot_data(seeds):
    mle_losses = []
    moment_based_losses = []

    for seed in seeds:
        mle_losses.append(np.load(f"data/mle_{seed}.npy"))
        moment_based_losses.append(np.load(f"data/moment_based_{seed}.npy"))

    mle_mean_losses, mle_std_losses = np.mean(mle_losses, axis=0), np.std(mle_losses, axis=0)
    moment_based_mean_losses, moment_based_std_losses = np.mean(moment_based_losses, axis=0), np.std(moment_based_losses, axis=0)

    plt.clf()
    xrange = np.arange(mle_mean_losses.shape[0])
    
    for model_idx in range(mle_mean_losses.shape[1]):
        color = np.random.rand(3)
        plt.plot(xrange, mle_mean_losses[:, model_idx], color=color, label=f"Good Model {model_idx}" if model_idx > 0 else f"Bad Model {model_idx}")
        # plt.fill_between(xrange, mle_mean_losses[:, model_idx] - mle_std_losses[:, model_idx], mle_mean_losses[:, model_idx] + mle_std_losses[:, model_idx], color=color, alpha=0.2)

    plt.xlabel("Number of Iterations")
    plt.ylabel("Classification loss on data collected")
    plt.legend()
    plt.grid(True)
    plt.savefig("mle_widetree_exp.png")

    plt.clf()    
    for model_idx in range(moment_based_mean_losses.shape[1]):
        color = np.random.rand(3)
        plt.plot(xrange, moment_based_mean_losses[:, model_idx], color=color, label=f"Good Model {model_idx}" if model_idx > 0 else f"Bad Model {model_idx}")
        # plt.fill_between(xrange, moment_based_mean_losses[:, model_idx] - moment_based_std_losses[:, model_idx], moment_based_mean_losses[:, model_idx] + moment_based_std_losses[:, model_idx], color=color, alpha=0.2)

    plt.xlabel("Number of Iterations")
    plt.ylabel("Value Moment loss on data collected")
    plt.legend()
    plt.grid(True)
    plt.savefig("moment_based_widetree_exp.png")


if __name__ == "__main__":
    seeds = range(10)
    plot_data(seeds)
    