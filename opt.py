# Fog
beta = [1, 2, 3, 4, 5]
intensity = [
    # 1,
    2,
    # 3,
    # 4,
    # 5
]

cloud_brightness = 0.5          # Heterogeneous Fog Brightness: 0.8 = 80% (available from 30% to 100%)

fog_types = [
    # "homo",
    "hetero"
]

img_shape = (1920, 1080)

# Atmospheric Light
sky_threshold = 0.01             # "sky"-pixels threshold (reduction from the maximal depth)
dark_channel_patch = 15         # patch size for dark channel calculation
dark_channel_top = 0.10         # percent of top brightest pixels: 0.05 = 5%

# Visualization
plot_turbulence_map = False
