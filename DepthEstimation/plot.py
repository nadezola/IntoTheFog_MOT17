import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def inverse_and_depth_maps(inv_depth, depth, output_path, file_name):
    if not output_path.exists():
        output_path.mkdir(parents=True)

    x = np.arange(depth.shape[1])
    y = np.arange(depth.shape[0])
    xx, yy = np.meshgrid(x, y)

    fig = plt.figure(figsize=plt.figaspect(0.5))

    # Depth subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(-xx, yy, -depth+60,  vmin=40., vmax=60., cmap=cm.coolwarm)
    #ax.plot_surface(-xx, yy, depth,  cmap=cm.coolwarm)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('Metric Depth')
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    ax.azim = -45
    ax.elev = 45

    # Invert depth subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(-xx, yy, inv_depth, cmap=cm.coolwarm)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('Aligned normalized MiDaS inverse depth')
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    ax.azim = -45
    ax.elev = 45

    fig.savefig(str(output_path / file_name) + '.png')
    plt.close(fig)