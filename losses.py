import numpy as np


def metric_relative_difference(gt, pred, valid_depths):
    result = 0.0
    sum_relative_difference_total = 0.0
    nr_valid_pixel_total = 0.0    

    gt = gt.reshape(-1)
    pred = pred.reshape(-1)
    valid_depths = valid_depths.reshape(-1)

    pred = pred * valid_depths
    target = gt * valid_depths
    difference = pred-target
    processed_difference = difference**2
    relative_difference = np.divide(processed_difference, target + 1e-5)

    sum_relative_difference = np.sum(relative_difference)
    nr_valid_pixel_total = np.sum(valid_depths)

    print("relative_difference: ", sum_relative_difference)
    print("nr_valid_pixels: ", nr_valid_pixel_total)

    result = sum_relative_difference/nr_valid_pixel_total

    return result