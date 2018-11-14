import numpy as np

def depth_to_logdepth(depth):
    return np.log(depth + 1e-5)

def logdepth_to_depth(logdepth):
    return np.exp(logdepth) - 1e-5

def metric_RMSE(gt, pred, valid_pixels, metric_type):
    result = 0.0
    sum_squared_difference = 0.0
    nr_valid_pixel_total = 0.0    

    gt = gt.reshape(-1)
    pred = pred.reshape(-1)
    valid_pixels = valid_pixels.reshape(-1)

    if metric_type == "linear":
        pred = pred * valid_pixels
    elif metric_type == "log":
        gt = depth_to_logdepth(gt)
        pred = depth_to_logdepth(pred)
        pred = pred * valid_pixels

    target = gt * valid_pixels

    difference = pred-target
    processed_difference = difference**2

    sum_squared_difference = np.sum(processed_difference)
    nr_valid_pixel_total = np.sum(valid_pixels)

    result = sum_squared_difference/nr_valid_pixel_total

    return result

def relative_difference_metric(gt, pred, valid_pixels, metric_type):
    result = 0.0
    sum_relative_difference = 0.0
    nr_valid_pixel_total = 0.0

    gt = gt.reshape(-1)
    pred = pred.reshape(-1)
    valid_pixels = valid_pixels.reshape(-1)

    pred = pred * valid_pixels
    target = gt * valid_pixels

    difference = pred-target

    if metric_type == "squared":
        processed_difference = difference**2
    elif metric_type == "abs":
        processed_difference = np.absolute(difference)

    relative_difference = np.divide(processed_difference, target + 1e-5) # processed_difference / (target + 1e-5) #
    sum_relative_difference = np.sum(relative_difference)
    nr_valid_pixel_total = np.sum(valid_pixels)

    result = sum_relative_difference / nr_valid_pixel_total
    return result