from tqdm import tqdm
import cv2


def calculate_seq_intensity(image_list):
    seq_mean, seq_std = 0, 0
    for i, f in tqdm(enumerate(image_list), total=len(image_list),
                     desc='Calculating Atmospheric Light for Sequence'):
        image = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) / 255.0
        mean, std = cv2.meanStdDev(image)
        seq_mean += mean
        seq_std += std

    atm_light = (seq_mean + 2 * seq_std) / len(image_list)

    print(f'Sequence Intensity: mean = {seq_mean} | std: {seq_std}')
    print(f"Atmospheric Light = {atm_light}")

    return atm_light
