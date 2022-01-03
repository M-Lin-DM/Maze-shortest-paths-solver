import matplotlib.pyplot as plt
import numpy as np


def plot_grayscale_img(array):
    plt.figure(figsize=(8, 8))
    ax = plt.axes()
    img = ax.imshow(array, cmap="gray")
    plt.show()


def contour(T, contour_interval, max_height, ax, line_color="white"):
    """Add contour plot of distance field T to current plot"""

    XX, YY = np.meshgrid(np.arange(0, T.shape[1]), np.arange(0, T.shape[0]))
    contour_levels = np.arange(0, max_height, contour_interval)
    # fig3, ax3 = plt.subplots(figsize=(8, 8))
    cset1 = ax.contourf(XX, YY, T, contour_levels, cmap="inferno", origin='upper')
    cset2 = ax.contour(XX, YY, T, contour_levels, colors=line_color, origin='upper', linewidths=0.5)


def quiver(ax, mu, interval):
    """Add quiver plot of distance field gradient mu to current plot"""

    margin_q = 3
    XX, YY = np.meshgrid(np.arange(0, mu.shape[1]), np.arange(0, mu.shape[0]))
    q = ax.quiver(XX[margin_q:-margin_q:interval, margin_q:-margin_q:interval],
                  YY[margin_q:-margin_q:interval, margin_q:-margin_q:interval],
                  mu[0, margin_q:-margin_q:interval, margin_q:-margin_q:interval],
                  mu[1, margin_q:-margin_q:interval, margin_q:-margin_q:interval],
                  color='black')


def plot_optimal_path(T, traj, contour_interval=None, line_color="white"):
    """ Plot distance field contour along with the path in traj"""

    a = 7
    max_height = np.max(T)
    fig3, ax3 = plt.subplots(figsize=(a, a))
    if contour_interval is not None:
        contour(T, contour_interval, max_height, ax3, line_color)
    else:
        contour_interval = int((np.max(T.shape) / 90))
        contour(T, contour_interval, max_height, ax3, line_color)

    # plt.title('Maze Solution', fontsize=15)

    ax3.plot(traj[:, 0], traj[:, 1], label='traj', color='blue', marker='.', ms=2)
    ax3.scatter(traj[:, 0], traj[:, 1], marker='.', facecolors='blue')

    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])

    # plt.tight_layout()

    plt.gca().set_aspect('equal')  # turn this off if using make_axes_locatable for the colorbar
    plt.show()
