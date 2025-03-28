# Fog
#beta = [0.06, 0.15, 0.3, 1]     # Visibility less than [50m, 20m, 10m, 3m]
# beta = [0.005, 0.01, 0.015, 0.03]     # Visibility less than [500m, 300m, 200m, 100m]
beta = [2, 3, 4, 5]
intensity = [1, 2, 3, 4]
cloud_brightness = 0.5          # Heterogeneous Fog Brightness: 0.8 = 80% (available from 30% to 100%)

img_shape = (607, 1080)

# Atmospheric Light
sky_threshold = 0.01             # "sky"-pixels threshold (reduction from the maximal depth)
dark_channel_patch = 15         # patch size for dark channel calculation
dark_channel_top = 0.10         # percent of top brightest pixels: 0.05 = 5%

# Visualization
save_depth_gray = False
save_depth_color = False
plot_metric_depth = False
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
