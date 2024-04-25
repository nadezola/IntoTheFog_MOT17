import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def inverse_and_metric_depths(inv_depth, metric_depth, output_path, file_name):
    if not output_path.exists():
        output_path.mkdir(parents=True)

    x = np.arange(metric_depth.shape[1])
    y = np.arange(metric_depth.shape[0])
    xx, yy = np.meshgrid(x, y)

    fig = plt.figure(figsize=plt.figaspect(0.5))

    # Invert depth subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(xx, yy, (1-inv_depth), vmin=0.15, vmax=0.9, cmap=cm.coolwarm_r)
    ax.set_zlabel('Predicted Depth', fontsize=10, labelpad=-4)
    ax.tick_params(labelsize=6, pad=-2)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.view_init(azim=30, elev=-150, vertical_axis='y')
    ax.set_zlim([0, 1])

    # Metric Depth subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    if metric_depth.max() > 50:
        metric_depth[metric_depth > 50] = 50
        ax.plot_surface(xx, yy, metric_depth, vmax=10, cmap=cm.coolwarm_r)
        ax.set_zlim([0, 50])
    else:
        ax.plot_surface(-xx, yy, -metric_depth, cmap=cm.coolwarm)
    ax.tick_params(labelsize=6, pad=-2)
    ax.set_zlabel('Metric Depth', fontsize=10, labelpad=-4)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.view_init(azim=30, elev=-150, vertical_axis='y')
    fig.savefig(str(output_path / file_name) + '.png')
    plt.close(fig)
