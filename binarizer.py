import numpy as np
import cv2

def improved_binarizer(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    total_pixels = image.size
    sum_total = 0
    for i in range(256):
        sum_total += i * histogram[i]
    sum_background, weight_background = 0, 0
    max_variance, optimal_threshold = 0, 0

    for t in range(256):
        weight_background += histogram[t]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * histogram[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance_between > max_variance:
            max_variance = variance_between
            optimal_threshold = t

    mask_1 = (image > optimal_threshold)
    mask_2 = ~mask_1

    if np.sum(mask_1) > np.sum(mask_2):
        image[mask_1] = 0
        image[mask_2] = 255
    else:
        image[mask_1] = 255
        image[mask_2] = 0

    return image

