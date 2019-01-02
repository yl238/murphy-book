import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import seaborn as sns

sns.set(style='white', font_scale=1.2)

def plot_gaussian_mixture():
    """
    This simple script plots the contour of a mixture model
    with three Gaussians and its 3D representation.
    """
    mu = [[0.22, 0.45], [0.5, 0.5], [0.77, 0.55]]
    sigma2 = [[0.018, 0.01],[0.011, 0.01]]
    sigma = [[0.011, -0.01], [-0.01, 0.018]]
    sigma3 = sigma

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y

    # Compute individual base distribution
    rv = multivariate_normal(mu[0], sigma)
    rv2 = multivariate_normal(mu[1], sigma2)
    rv3 = multivariate_normal(mu[2], sigma)

    # Visualise on 2D
    f, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].contour(X, Y, rv.pdf(pos), colors='r')
    axes[0].contour(X, Y, rv2.pdf(pos), colors='g')
    axes[0].contour(X, Y, rv3.pdf(pos), colors='b')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0.17, 0.85)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)

    # Plot 3D sum of the mixture
    Z = rv2.pdf(pos) + rv.pdf(pos) + rv3.pdf(pos)
    axes[1] = plt.subplot(122, projection='3d')
    axes[1].plot_surface(X, Y, Z,rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
    axes[1].view_init(20, 150)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0.15, 0.85)
    axes[1].set_axis_off()
   