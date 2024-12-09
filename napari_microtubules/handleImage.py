import numpy as np
from napari_microtubules.handleTubuleSelection import *


class LoadTIFF:
    def __init__(self, video: np.array):
        """ load TIFF file """
        self.tiff_video = video
        print(type(self.tiff_video))

        # store number of frames
        self.tiff_frames_num = video.shape[0]

        # create images array with NumPy slicing
        self.tiff_image = video

        # convert to grayscale
        self.tiff_gray_image = video // 257


def custom_connected_components(binary_image, connectivity=8, min_size=50):
    """ connected components function without using cv.connectedComponents """
    binary_image = (binary_image > 0).astype(np.uint8)
    height, width = binary_image.shape

    # output labels matrix initialized to 0
    labels = np.zeros_like(binary_image, dtype=np.int32)
    label = 0

    # define neighbor for 4- or 8-connectivity
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        raise ValueError("connectivity must be either 4 or 8.")

    # do flood-fill
    def flood_fill(x, y, current_label):
        stack = [(x, y)]
        component_size = 0
        while stack:
            cx, cy = stack.pop()
            if labels[cx, cy] == 0 and binary_image[cx, cy] == 1:
                labels[cx, cy] = current_label
                component_size += 1
                for dx, dy in neighbors:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < height and 0 <= ny < width and labels[nx, ny] == 0:
                        stack.append((nx, ny))
        return component_size

    # iterate through all pixels to find connected components
    label_sizes = []
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 1 and labels[i, j] == 0:
                label += 1
                size = flood_fill(i, j, label)
                label_sizes.append(size)

    # filter components by size
    label_sizes = np.array(label_sizes)
    large_labels = np.where(label_sizes >= min_size)[0] + 1
    filtered_labels = np.isin(labels, large_labels).astype(np.uint8) * 255

    return label, labels, filtered_labels


def crop_image(image, x1, x2, y1, y2):
    """ crop image based on input size """
    mask = np.zeros(image.shape)
    mask[x1:x2, y1:y2] = 1
    return mask * image


def track_microtubule(image, user_line, smoothing_kernel, intensity_threshold, padding=5, connectivity=8):
    """ track connected components of microtubules based on user-defined line input """
    point_start = [round(user_line[0][1]), round(user_line[0][2])]
    point_end = [round(user_line[1][1]), round(user_line[1][2])]

    # Denoise and preprocess image
    contrast_image, processed_image = denoise_image(image)

    # Get region of interest (ROI) and perform intensity thresholding
    row_min = max(min(point_start[0], point_end[0]) - padding, 0)
    row_max = min(max(point_start[0], point_end[0]) + padding, image.shape[0])
    col_min = max(min(point_start[1], point_end[1]) - padding, 0)
    col_max = min(max(point_start[1], point_end[1]) + padding, image.shape[1])

    roi = processed_image[row_min:row_max, col_min:col_max]
    total_intensity = np.float64(0)
    valid_pixel_count = np.float64(0)

    # Calculate based on ROI intensity
    for row in range(roi.shape[0]):
        for col in range(roi.shape[1]):
            if roi[row, col] > 150:
                valid_pixel_count += 1
                total_intensity += roi[row, col]

    adaptive_threshold = (total_intensity / valid_pixel_count) * intensity_threshold

    # Do binary thresholding
    binary_image = do_adaptive_thresholding(processed_image)
    thresholded_image = do_binary_thresholding(contrast_image, adaptive_threshold - 5).astype(np.uint8)
    thresholded_image = normal_opening(thresholded_image, 2)

    # Combine thresholded images
    combined_binary = binary_image * (thresholded_image / 255)
    cropped_binary = crop_image(combined_binary,
                                max(row_min - 100, 0), min(row_max + 100, combined_binary.shape[0]),
                                max(col_min - 100, 0), min(col_max + 100, combined_binary.shape[1]))

    final_binary = normal_opening(cropped_binary, 2).astype(np.uint8)
    # final_binary = otsu_threshold(cropped_binary).astype(np.uint8)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Binarized Image")
    # plt.imshow(final_binary, cmap='gray')
    # plt.axis('off')

    # Find connected components
    num_labels, labels, filtered_labels = custom_connected_components(final_binary, connectivity)

    # get regions close to the line
    def calculate_distance_to_line(point, point_start, point_end):
        """ calculate perpendicular distance from a point to a line """
        # convert points to numpy arrays
        line_vector = np.array(point_end) - np.array(point_start)
        point_vector = np.array(point) - np.array(point_start)

        # compute the norm
        distance = abs(np.cross(line_vector, point_vector) / np.linalg.norm(line_vector))
        return distance

    # helper function to find labels within a specific distance to the line
    def find_labels_near_line(row_min, row_max, col_min, col_max, distance_threshold, binary_image, labels, point_start, point_end):
        """ find labels of components near a line within a given distance """
        selected = set()
        # iterate over defined region of binary image
        for row in range(max(row_min, 0), min(row_max, binary_image.shape[0])):
            for col in range(max(col_min, 0), min(col_max, binary_image.shape[1])):
                # if the pixel is part of a component
                if binary_image[row, col] != 0:
                    point = [row, col]
                    # calculate the distance to line
                    distance = calculate_distance_to_line(point, point_start, point_end)
                    if distance < distance_threshold:
                        # add label of the component if it's near the line
                        selected.add(labels[row, col])
        return selected

    selected_labels = find_labels_near_line(
        row_min + 3, row_max - 3, col_min + 3, col_max - 3, 
        distance_threshold=5, 
        binary_image=final_binary, 
        labels=labels, 
        point_start=point_start, 
        point_end=point_end
    )

    # if no labels are found near the line, extend the search region
    if not selected_labels:
        selected_labels = find_labels_near_line(
            row_min - 5, row_max + 5, col_min - 5, col_max + 5, 
            distance_threshold=15, 
            binary_image=final_binary, 
            labels=labels, 
            point_start=point_start, 
            point_end=point_end
        )

    # Filter the binary image for selected labels
    final_binary = final_binary * np.isin(labels, list(selected_labels)).astype(np.uint8)
    final_binary = normal_closing(final_binary, 2)

    # Select the best line from the filtered microtubule components
    line_info = select_best_line(final_binary, point_start, point_end)
    if line_info is None:
        return

    [[y1, x1, y2, x2]], line_derivative, hough_lines, hough_line = line_info
    refined_binary = opening(final_binary, smoothing_kernel, line_derivative)

    refined_start = np.array([x1, y1])
    refined_end = np.array([x2, y2])

    # calculate the center and length of the line
    line_center = (refined_start + refined_end) // 2
    line_direction = refined_end - refined_start
    line_length = np.linalg.norm(line_direction)

    # calculate cross product norm for the line
    line_norm = np.linalg.norm(line_direction)

    # get bounds for rows and columns
    row_min = max(row_min - 105, 0)
    row_max = min(row_max + 105, refined_binary.shape[0])
    col_min = max(col_min - 105, 0)
    col_max = min(col_max + 105, refined_binary.shape[1])

    # go through through each pixel in the region
    for row in range(row_min, row_max):
        for col in range(col_min, col_max):
            if refined_binary[row, col] != 0:
                point = np.array([row, col])

                # calculate distance to the line using cross product
                distance_to_line = abs(np.cross(line_direction, point - refined_start) / line_norm)

                # calculate distance to the line center
                distance_to_center = np.linalg.norm(point - line_center)

                # remove outliers based on distance thresholds
                if distance_to_line > 5 or distance_to_center > (6 / 11) * line_length:
                    refined_binary[row, col] = 0

    refined_binary = normal_closing(refined_binary, 8)
    refined_binary = normal_opening(refined_binary, 2)

    return [refined_start, refined_end], refined_binary.astype(np.uint8), line_length


# MORPHOLOGICAL FUNCTIONS / OTHER HELPERS
from scipy.ndimage import gaussian_filter

def do_binary_thresholding(image, threshold_value):
    """ apply binary thresholding on input image """
    _, binary_image = cv.threshold(image, threshold_value, image.max(), cv.THRESH_BINARY)
    return binary_image


def do_adaptive_thresholding(image):
    """ apply adaptive thresholding on input image """
    return cv.adaptiveThreshold(image, image.max(), cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)


def gaussian_blur(image, sigma=2):
    """ do Gaussian blur on input image """
    return gaussian_filter(image, sigma=sigma)


def denoise_image(image):
    """ denoise input image and boost constrast """
    blurred_image = gaussian_blur(image.astype(np.uint8), 5)
    contrast_enhancer = cv.createCLAHE(clipLimit=None, tileGridSize=(8, 8))
    enhanced_image = contrast_enhancer.apply(blurred_image)

    enhanced_original_image = contrast_enhancer.apply(image.astype(np.uint8))
    return enhanced_image, enhanced_original_image


def shift(arr, num, fill_value=np.nan):
    """ shift elements of input array by some input positions """
    result = np.full_like(arr, fill_value)
    if num > 0:
        result[num:] = arr[:-num]
    elif num < 0:
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def create_kernel(kernel_input, distance_input):
    """ make a custom kernel """
    kernel = np.zeros((kernel_input, kernel_input), np.uint8)
    if -1 < distance_input < 1:
        offset = round((kernel_input - distance_input * kernel_input + 1) / 2)
        for j in range(kernel_input):
            mid = round(j * distance_input) + offset
            low, high = max(0, round(mid - 1)), min(kernel_input, round(mid))
            for i in range(low, high):
                kernel[i, j] = 1
    else:
        distance_input = 1 / distance_input
        offset = round((kernel_input - distance_input * kernel_input + 1) / 2)
        for i in range(kernel_input):
            mid = round(i * distance_input) + offset
            low, high = max(0, round(mid - 1)), min(kernel_input, round(mid))
            for j in range(low, high):
                kernel[i, j] = 1
    return kernel


def adjust_kernel(kernel, kernel_size, distance_from_center):
    """ Adjust kernel to make sure the boundary is aligned correctly. """
    if -1 < distance_from_center < 1:
        top_boundary, bottom_boundary = np.argmax(kernel[:, 0] == 1), np.argmax(kernel[::-1, -1] == 1)
        if top_boundary - bottom_boundary > 1:
            kernel[:, :] = np.roll(kernel, -1, axis=0)
        elif bottom_boundary - top_boundary > 1:
            kernel[:, :] = np.roll(kernel, 1, axis=0)
    else:
        left_boundary, right_boundary = np.argmax(kernel[-1] == 1), np.argmax(kernel[0] == 1)
        if left_boundary - right_boundary > 1:
            kernel[:] = np.roll(kernel, -1, axis=1)
        elif right_boundary - left_boundary > 1:
            kernel[:] = np.roll(kernel, 1, axis=1)
    return kernel


def closing(img_bin, k, d):
    """ do morphological closing on binary input image using the kernel """
    kernel = create_kernel(k, d)
    kernel = adjust_kernel(kernel, k, d)
    return cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)


def opening(img_bin, k, d):
    """ do morphological opening on binary input image using the kernel """
    kernel = create_kernel(k, d)
    kernel = adjust_kernel(kernel, k, d)
    return cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)


def normal_opening(img_bin, k):
    """ do normal opening on binary image (not using the kernel adjustment method like above) """
    kernel = np.ones((k, k), np.uint8)
    return cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)


def normal_closing(img_bin, k):
    """ do normal closing on binary image (not using the kernel adjustment method like above) """
    kernel = np.ones((k, k), np.uint8)
    return cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)