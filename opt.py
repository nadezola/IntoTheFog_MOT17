# Fog
beta = [0.06, 0.15, 0.3, 1]     # Visibility less than [50m, 20m, 10m, 3m]
intensity = [1, 2, 3, 4]
cloud_brightness = 0.8          # Heterogeneous Fog Brightness: 0.8 = 80% (available from 30% to 100%)

# Atmospheric Light
sky_threshold = 0.95            # "sky"-pixels are greater than this threshold
dark_channel_patch = 10         # patch size for dark channel calculation
dark_channel_top = 0.05         # percent of top brightest pixels: 0.05 = 5%

# Visualization
save_depth_gray = True
save_depth_color = True
plot_metric_depth = True
plot_turbulence_map = True


# 3D Reference Points
seq_info = {
    "MOT17-02": {
        "min_dist": 1.5,
        "max_dist": 1e6,            # Sky presents
    },
    "MOT17-04": {
        "min_dist": 4.5,
        "max_dist": 20,
    },
    "MOT17-05": {
        "min_dist": 1.5,
        "max_dist": 1e6,
    },
    "MOT17-09": {
        "min_dist": 1.5,
        "max_dist": 15,
    },
    "MOT17-10": {
        "min_dist": 1.5,
        "max_dist": 60,
    },
    "MOT17-11": {
        "min_dist": 1.5,
        "max_dist": 40,
    },
    "MOT17-13": {
        "min_dist": 2,
        "max_dist": 60,
    },
}
