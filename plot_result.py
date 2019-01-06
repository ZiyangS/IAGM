import numpy as np
import matplotlib.pyplot as plt

def plot_result(Samp, X, c, n, outfile, Ngrid=100, M=4):
    """Plots samples drawn from the posterior"""
    fig, ax = plt.subplots()

    samples = Samp[-1]
    Xj = [X[np.where(c == j), :] for j, nj in enumerate(n)]
    for index in range(samples.M):
        if (Xj[index][0].shape)[0] == 1:
            ax.scatter(np.squeeze(Xj[index])[0], np.squeeze(Xj[index])[1], alpha=0.2)
        else:
            ax.scatter(np.squeeze(Xj[index])[:, 0], np.squeeze(Xj[index])[:, 1],  alpha=0.2)
    fig.savefig(outfile)

