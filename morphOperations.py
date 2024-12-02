import cv2 as cv
import numpy as np


def crop_image(image, x1, x2, y1, y2):
    """Crop image based on input size."""
    mask = np.zeros(image.shape)
    mask[x1:x2, y1:y2] = 1
    return mask * image


def do_binary_thresholding(image, threshold_value):
    """Apply binary thresholding to the image."""
    _, binary_image = cv.threshold(image, threshold_value, image.max(), cv.THRESH_BINARY)
    return binary_image


def do_adaptive_thresholding(image):
    """Apply adaptive thresholding to the image."""
    return cv.adaptiveThreshold(image, image.max(), cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)


def denoise_image(image):
    """Denoise the image and enhance contrast."""
    blurred_image = cv.medianBlur(image.astype(np.uint8), 5)
    contrast_enhancer = cv.createCLAHE(clipLimit=None, tileGridSize=(8, 8))
    enhanced_image = contrast_enhancer.apply(blurred_image)

    enhanced_original_image = contrast_enhancer.apply(image.astype(np.uint8))
    return enhanced_image, enhanced_original_image


def shift(arr, num, fill_value=np.nan):
    """Shift elements of an array by a specified number of positions."""
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


def create_kernel(k, d):
    """Create a custom kernel based on the size k and parameter d."""
    kernel = np.zeros((k, k), np.uint8)
    if -1 < d < 1:
        offset = round((k - d * k + 1) / 2)
        for j in range(k):
            mid = round(j * d) + offset
            low, high = max(0, round(mid - 1)), min(k, round(mid))
            for i in range(low, high):
                kernel[i, j] = 1
    else:
        d = 1 / d
        offset = round((k - d * k + 1) / 2)
        for i in range(k):
            mid = round(i * d) + offset
            low, high = max(0, round(mid - 1)), min(k, round(mid))
            for j in range(low, high):
                kernel[i, j] = 1
    return kernel


def adjust_kernel(kernel, k, d):
    """Adjust the kernel to ensure proper boundary alignment."""
    if -1 < d < 1:
        up, down = np.argmax(kernel[:, 0] == 1), np.argmax(kernel[::-1, -1] == 1)
        if up - down > 1:
            kernel[:, :] = np.roll(kernel, -1, axis=0)
        elif down - up > 1:
            kernel[:, :] = np.roll(kernel, 1, axis=0)
    else:
        front, back = np.argmax(kernel[-1] == 1), np.argmax(kernel[0] == 1)
        if front - back > 1:
            kernel[:] = np.roll(kernel, -1, axis=1)
        elif back - front > 1:
            kernel[:] = np.roll(kernel, 1, axis=1)
    return kernel


def closing(img_bin, k, d):
    """Perform closing morphological operation on binary image."""
    kernel = create_kernel(k, d)
    kernel = adjust_kernel(kernel, k, d)
    return cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)


def opening(img_bin, k, d):
    """Perform opening morphological operation on binary image."""
    kernel = create_kernel(k, d)
    kernel = adjust_kernel(kernel, k, d)
    return cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)


def normal_closing(img_bin, k):
    """Perform normal closing on binary image."""
    kernel = np.ones((k, k), np.uint8)
    return cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)


def normal_opening(img_bin, k):
    """Perform normal opening on binary image."""
    kernel = np.ones((k, k), np.uint8)
    return cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
