import numpy as np
from napari_microtubules.morphOperations import *
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


def crop_image(image, x1, x2, y1, y2):
    """ crop image based on input size """
    mask = np.zeros(image.shape)
    mask[x1:x2, y1:y2] = 1
    return mask * image


def crop_image(image, x1, x2, y1, y2):
    """ crop image based on input size """
    mask = np.zeros(image.shape)
    mask[x1:x2, y1:y2] = 1
    return mask * image


def custom_connected_components(binary_image, connectivity=8, min_size=50):
    """ connected components function without using cv.connectedComponents """
    binary_image = (binary_image > 0).astype(np.uint8)
    height, width = binary_image.shape

    # output labels matrix initialized to 0
    labels = np.zeros_like(binary_image, dtype=np.int32)
    label = 0

    # define neighbor offsets for 4- or 8-connectivity
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


def track_microtubule(image, user_line, smoothing_kernel, intensity_threshold, connectivity=8):
    """ track connected components of microtubules based on user-defined line input """
    point_start = [round(user_line[0][1]), round(user_line[0][2])]
    point_end = [round(user_line[1][1]), round(user_line[1][2])]

    # Denoise and preprocess image
    contrast_image, processed_image = denoise_image(image)

    # Get region of interest (ROI) and perform intensity thresholding
    row_min = max(min(point_start[0], point_end[0]) - 5, 0)
    row_max = min(max(point_start[0], point_end[0]) + 5, image.shape[0])
    col_min = max(min(point_start[1], point_end[1]) - 5, 0)
    col_max = min(max(point_start[1], point_end[1]) + 5, image.shape[1])

    roi = processed_image[row_min:row_max, col_min:col_max]
    total_intensity = np.float64(0)
    valid_pixel_count = np.float64(0)

    # Calculate adaptive threshold based on ROI intensity
    for row in range(roi.shape[0]):
        for col in range(roi.shape[1]):
            if roi[row, col] > 150:
                valid_pixel_count += 1
                total_intensity += roi[row, col]

    adaptive_threshold = (total_intensity / valid_pixel_count) * intensity_threshold

    # Perform binary thresholding
    binary_image = do_adaptive_thresholding(processed_image)
    thresholded_image = do_binary_thresholding(contrast_image, adaptive_threshold - 5)
    thresholded_image = thresholded_image.astype(np.uint8)
    thresholded_image = normal_opening(thresholded_image, 2)

    # Combine thresholded images
    combined_binary = binary_image * (thresholded_image / 255)
    cropped_binary = crop_image(combined_binary,
                                max(row_min - 100, 0), min(row_max + 100, combined_binary.shape[0]),
                                max(col_min - 100, 0), min(col_max + 100, combined_binary.shape[1]))

    final_binary = normal_opening(cropped_binary, 2).astype(np.uint8)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Binarized Image")
    # plt.imshow(final_binary, cmap='gray')
    # plt.axis('off')

    # Find connected components
    num_labels, labels, filtered_labels = custom_connected_components(final_binary, connectivity)

    # Get regions close to the user-defined line
    selected_labels = set()
    for row in range(max(row_min + 3, 0), min(row_max - 3, final_binary.shape[0])):
        for col in range(max(col_min + 3, 0), min(col_max - 3, final_binary.shape[1])):
            if final_binary[row, col] != 0:
                point = np.array([row, col])
                distance_to_line = abs(
                    np.cross(np.array(point_end) - np.array(point_start), point - np.array(point_start))
                    / np.linalg.norm(np.array(point_end) - np.array(point_start))
                )
                if distance_to_line < 5:
                    selected_labels.add(labels[row, col])

    # If no components are found near the line, extend the search region
    if not selected_labels:
        for row in range(max(row_min - 5, 0), min(row_max + 5, final_binary.shape[0])):
            for col in range(max(col_min - 5, 0), min(col_max + 5, final_binary.shape[1])):
                if final_binary[row, col] != 0:
                    point = np.array([row, col])
                    distance_to_line = abs(
                        np.cross(np.array(point_end) - np.array(point_start), point - np.array(point_start))
                        / np.linalg.norm(np.array(point_end) - np.array(point_start))
                    )
                    if distance_to_line < 15:
                        selected_labels.add(labels[row, col])

    # Filter the binary image for selected labels
    selected_labels = list(selected_labels)
    filtered_labels = np.isin(labels, selected_labels).astype(np.uint8)
    final_binary = final_binary * filtered_labels

    # Smooth the final image
    final_binary = normal_closing(final_binary, 2)

    # Select the best line from the filtered microtubule components
    line_info = select_best_line(final_binary, point_start, point_end)
    if line_info is None:
        return

    [[y1, x1, y2, x2]], line_derivative, hough_lines, hough_line = line_info
    refined_binary = opening(final_binary, smoothing_kernel, line_derivative)

    refined_start = np.array([x1, y1])
    refined_end = np.array([x2, y2])

    # Calculate the center and length of the line
    line_center = (refined_start + refined_end) // 2
    line_length = np.linalg.norm(refined_end - refined_start)

    # Remove outliers based on distance from the line and center
    for row in range(max(row_min - 105, 0), min(row_max + 105, refined_binary.shape[0])):
        for col in range(max(col_min - 105, 0), min(col_max + 105, refined_binary.shape[1])):
            if refined_binary[row, col] != 0:
                point = np.array([row, col])
                distance_to_line = abs(np.cross(refined_end - refined_start, point - refined_start)
                                       / np.linalg.norm(refined_end - refined_start))
                distance_to_center = np.linalg.norm(point - line_center)
                if distance_to_line > 5 or distance_to_center > (6 / 11) * line_length:
                    refined_binary[row, col] = 0

    refined_binary = normal_closing(refined_binary, 8)
    refined_binary = normal_opening(refined_binary, 2)

    return [refined_start, refined_end], refined_binary.astype(np.uint8), line_length
