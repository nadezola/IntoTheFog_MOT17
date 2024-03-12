"""
    beta values: [0.015, 0.03, 0.04, 0.06, 0.15]
    regulate fog thickness and
    correspond to visibility less than:
    [200m, 100m, 75m, 50m, 20m]

    Formula: beta = -ln(0.05) / visibility
"""
beta = [0.015, 0.03, 0.04, 0.06, 0.15]
visibility = [1, 2, 3, 4, 5]     # [200m, 100m, 75m, 50m, 20m]

# Depth Estimation
estimate_depth = True
save_norm_inv_depth = True
plot_metric_depth = True

# Heterogeneous Fog
heterogeneous_fog = True
cloud_brightness = 0.5   # 0.5 = 50%; available range [0.3, 1.0]
plot_turbulence_map = False

# Sequence Info
seq_info = {
    "MOT17-02": {
        "min_dist": 1,
        "max_dist": 50,
        "horizon": True,
        "static_camera": True
    },
    "MOT17-04": {
        "min_dist": 1,
        "max_dist": 50,
        "horizon": False,
        "static_camera": True
    },
    "MOT17-05": {
        "min_dist": 1,
        "max_dist": 50,
        "horizon": True,
        "static_camera": False
    },
    "MOT17-09": {
        "min_dist": 1,
        "max_dist": 50,
        "horizon": False,
        "static_camera": True
    },
    "MOT17-10": {
        "min_dist": 1,
        "max_dist": 50,
        "horizon": False,
        "static_camera": False
    },
    "MOT17-11": {
        "min_dist": 1,
        "max_dist": 50,
        "horizon": False,
        "static_camera": False
    },
    "MOT17-13": {
        "min_dist": 1,
        "max_dist": 50,
        "horizon": False,
        "static_camera": False
    },
}
