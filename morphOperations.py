import cv2 as cv
import numpy as np


def cropImage(image, x1, x2, y1, y2):
    """ crop image based on input size """
    mask = np.zeros(image.shape)
    mask[x1:x2, y1:y2] = 1
    return mask * image


def doBinaryThresholding(image, threshold_value):
    """ do binary thresholding on input image """
    _, binary_image = cv.threshold(image, threshold_value, image.max(), cv.THRESH_BINARY)
    return binary_image


def doAdaptiveThresholding(image):
    """ do adaptive thresholding on input image """
    return cv.adaptiveThreshold(image, image.max(), cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)


def denoiseImage(image):
    """ denoise image and make contrast better """
    image = np.array(image.astype(np.uint8))
    blurred_image = cv.medianBlur(image, 5)
    denoised_image = np.array(blurred_image.astype(np.uint8))
    contrast_enhancer = cv.createCLAHE(clipLimit=None, tileGridSize=(8, 8))
    enhanced_image = contrast_enhancer.apply(denoised_image)

    original_image = np.array(image.astype(np.uint8))
    contrast_enhancer = cv.createCLAHE(clipLimit=None, tileGridSize=(8, 8))
    enhanced_original_image = contrast_enhancer.apply(original_image)
    return enhanced_image, enhanced_original_image


def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def closing(img_bin, k, d):
    kernel = np.zeros((k, k), np.uint8)
    offset = round((k - d * k + 1) / 2) if abs(d) < 1 else round((k - (1 / d) * k + 1) / 2)
    
    for i in range(k):
        for j in range(k):
            mid = round(i * d) + offset if abs(d) < 1 else round(j * (1 / d)) + offset
            kernel[i, j] = 1 if (0 <= mid < k) else 0

    # adjust kernel shift based on boundaries
    if abs(d) < 1:
        up, down = np.argmax(kernel[:, 0] == 1), np.argmax(kernel[::-1, -1] == 1)
        if up - down > 1: kernel[:, :] = np.roll(kernel, -1, axis=0)
        elif down - up > 1: kernel[:, :] = np.roll(kernel, 1, axis=0)
    else:
        front, back = np.argmax(kernel[-1] == 1), np.argmax(kernel[0] == 1)
        if front - back > 1: kernel[:] = np.roll(kernel, -1, axis=1)
        elif back - front > 1: kernel[:] = np.roll(kernel, 1, axis=1)

    return cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)


def opening(img_bin, k, d):
    print(d)
    kernel = np.zeros((k, k), np.uint8)
    if -1 < d < 1:
        offset = round((k - d * k + 1) / 2)
        for j in range(k):
            mid = round(j * d) + offset
            low, high = max(0, round(mid - 1)), min(k, round(mid))
            for i in range(low, high):
                kernel[i, j] = 1
        up, down = 0, 0
        for i in range(k):
            if kernel[i, 0] == 1:
                up = i
                break
        for i in range(k - 1, -1, -1):
            if kernel[i, -1] == 1:
                down = k - 1 - i
                break
        if up - down > 1:
            for j in range(k):
                kernel[:, j] = shift(kernel[:, j], -1, 0)
        elif down - up > 1:
            for j in range(k):
                kernel[:, j] = shift(kernel[:, j], 1, 0)
    else:
        d = 1 / d
        offset = round((k - d * k + 1) / 2)
        for i in range(k):
            mid = round(i * d) + offset
            low, high = max(0, round(mid - 1)), min(k, round(mid))
            for j in range(low, high):
                kernel[i, j] = 1
        front, back = 0, 0
        for j in range(k):
            if kernel[-1, j] == 1:
                front = j
                break
        for j in range(k - 1, -1, -1):
            if kernel[0, j] == 1:
                back = k - 1 - j
                break
        if front - back > 1:
            for i in range(k):
                kernel[i] = shift(kernel[i], -1, 0)
        elif back - front > 1:
            for i in range(k):
                kernel[i] = shift(kernel[i], 1, 0)
    print(kernel)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
    return img_bin


def normal_closing(img_bin, k):
    kernel = np.ones((k, k), np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)
    return img_bin


def normal_opening(img_bin, k):
    kernel = np.ones((k, k), np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
    return img_bin
