"""
===================================
Simple 1D Kernel Density Estimation
===================================
This example uses the class:sklearn.neighbors.KernelDensity class to 
demonstrate the pronciples of Kernel Density Estimation in one dimension.

1. The first plot shows one of the problems with using histograms to visualize the density
    of points in 1D.
2. All available kernels in scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


# 1.Plot the progression of histograms to kernels
np.random.seed(1)
N = 20
# bimodal distribution
x = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
x_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
bins = np.linspace(-5, 10, 10)

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# histogram 1
ax[0, 0].hist(x[:, 0], bins=bins, fc='#AAAAFF', normed=True)
ax[0, 0].text(-3.5, 0.31, "Histogram")

# histogram 2
ax[0, 1].hist(x[:, 0], bins=bins + 0.75, fc='#AAAAFF', normed=True)
ax[0, 1].text(-3.5, 0.31, "Histogram, bins shifted")

# Tophat KDE
kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(x)
log_den = kde.score_samples(x_plot)
ax[1, 0].fill(x_plot[:, 0], np.exp(log_den), fc='#AAAAFF')
ax[1, 0].text(-3.5, 0.31, "Tophat Kernel Density")

# Gaussian KDE
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(x)
log_den = kde.score_samples(x_plot)
print 'gaussian ll mean : %f' % np.mean(np.exp(log_den))
ax[1, 1].fill(x_plot[:, 0], np.exp(log_den), fc='#AAAAFF')
ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")

for axi in ax.ravel():
    axi.plot(x[:, 0], np.zeros(x.shape[0]) - 0.01, '+k')
    axi.set_xlim(-4, 9)
    axi.set_ylim(-0.02, 0.34)

for axi in ax[:, 0]:
    axi.set_ylabel('Normalized Density')

for axi in ax[1, :]:
    axi.set_xlabel('x')


plt.show()

# ---------------------------------------------------------------
# plot all available kernels
x_plot = np.linspace(-6, 6, 1000)[:, None]
x_src = np.zeros((1, 1))

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)

def format_func(x, loc):
    if x == 0:
        return '0'
    elif x == 1:
        return 'h'
    elif x == -1:
        return '-h'
    else:
        return '%ih' % x

for i, kernel in enumerate(['gaussian', 'tophat', 'epanechnikov',
                            'exponential', 'linear', 'cosine']):
    axi = ax.ravel()[i]
    log_dens = KernelDensity(kernel=kernel).fit(x_src).score_samples(x_plot)
    axi.fill(x_plot[:, 0], np.exp(log_dens), '-k', fc='#AAAAFF')
    axi.text(-2.6, 0.95, kernel)

    axi.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    axi.xaxis.set_major_locator(plt.MultipleLocator(1))
    axi.yaxis.set_major_locator(plt.NullLocator())

    axi.set_ylim(0, 1.05)
    axi.set_xlim(-2.9, 2.9)

ax[0, 1].set_title('Available Kernels')

plt.show()

# ---------------------------------------------------------------
# Plot a 1D density example
N = 100
np.random.seed(1)
x = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
x_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
true_dens = 0.3*norm(0, 1).pdf(x_plot[:, 0]) + 0.7*norm(5, 1).pdf(x_plot[:, 0])
fig, ax = plt.subplots()
ax.fill(x_plot[:, 0], true_dens, fc='black', alpha=0.2, label='input distribution')

for kernel in ['gaussian', 'tophat', 'epanechnikov']:
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(x)
    log_dens = kde.score_samples(x_plot)
    print 'kernel:%s, log_dens:%f' % (kernel, np.sum(np.exp(log_dens)))
    ax.plot(x_plot[:, 0], np.exp(log_dens), '-', label="kernel = '{0}'".format(kernel))

ax.text(6, 0.38, "N={0} points".format(N))
ax.legend(loc='upper left')
ax.plot(x[:, 0], -0.005 - 0.01 * np.random.random(x.shape[0]), '+k')

ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)
plt.show()