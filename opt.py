"""
    beta values: [0.015, 0.03, 0.04, 0.06, 0.15]
    regulate fog thickness and
    correspond to visibility less than:
    [200m, 100m, 75m, 50m, 20m]

    Formula: beta = -ln(0.05) / visibility
"""
beta = [0.015, 0.03, 0.04, 0.06, 0.15]
visibility = [200, 100, 75, 50, 20]     # [200m, 100m, 75m, 50m, 20m]

# Heterogeneous Fog
heterogeneous_fog = True
cloud_brightness = 0.5   # 0.5 = 50%; available range [0.3, 1.0]
plot_turbulence_map = False

# Metrics for Depth in [m]
metric_info = {
    "MOT17-02": {
        "min_dist": 1.5,
        "max_dist": 60
    },
    "MOT17-04": {
        "min_dist": 1.5,
        "max_dist": 60
    },
    "MOT17-05": {
        "min_dist": 1.5,
        "max_dist": 60
    },
    "MOT17-09": {
        "min_dist": 1.5,
        "max_dist": 60
    },
    "MOT17-10": {
        "min_dist": 1.5,
        "max_dist": 60
    },
    "MOT17-11": {
        "min_dist": 1.5,
        "max_dist": 60
    },
    "MOT17-13": {
        "min_dist": 1.5,
        "max_dist": 60
    },
}
