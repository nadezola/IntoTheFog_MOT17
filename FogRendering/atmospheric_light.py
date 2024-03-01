from tqdm import tqdm
import cv2

def horizon_intensity(image_list, depth_maps, horizon):
    seq_mean = 0
    for i, f in tqdm(enumerate(image_list), total=len(image_list),
                     desc='Calculating Atmospheric Light for Sequence'):
        image = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) / 255.0
        inf_pixels = image[(depth_maps[i] > horizon)]
        mean = cv2.mean(inf_pixels)[0]
        seq_mean += mean

    atm_light = seq_mean / len(image_list)
    print(f"Atmospheric Light = {atm_light}")

    return atm_light


def image_intensity(image_list):
    seq_mean, seq_std = 0, 0
    for i, f in tqdm(enumerate(image_list), total=len(image_list),
                     desc='Calculating Atmospheric Light for Sequence'):
        image = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) / 255.0
        mean, std = cv2.meanStdDev(image)
        seq_mean += mean
        seq_std += std

    atm_light = ((seq_mean + 2 * seq_std) / len(image_list)).mean()
    if atm_light > 1:
        atm_light = 1
        print(f"Atmospheric > 1")
    print(f"Atmospheric Light = {atm_light}")

    return atm_light
