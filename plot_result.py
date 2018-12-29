import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from numpy.linalg import inv,eig
from scipy.stats import norm


def plot_result(Samp, X, c, n, outfile, Ngrid=100, M=4):
    """Plots samples of ellipses drawn from the posterior"""
    fig, ax = plt.subplots()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    samples = Samp[-1]
    Xj = [X[np.where(c == j), :] for j, nj in enumerate(n)]

    for index in range(samples.M):
        ax.scatter(np.squeeze(Xj[index])[:, 0], np.squeeze(Xj[index])[:, 1], color=colors[index], alpha=0.2)
    plt.show()
    plt.savefig(outfile, dpi=300)

