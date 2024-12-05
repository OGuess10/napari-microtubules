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


def track_microtubule(image, user_line, smoothing_kernel, intensity_threshold):
    """Track target microtubule based on user-defined line input."""
    point_start = [round(user_line[0][1]), round(user_line[0][2])]
    point_end = [round(user_line[1][1]), round(user_line[1][2])]

    # denoise
    contrast_image, processed_image = denoise_image(image)

    # get region of interest and do intensity thresholding
    row_min = max(min(point_start[0], point_end[0]) - 5, 0)
    row_max = min(max(point_start[0], point_end[0]) + 5, image.shape[0])
    col_min = max(min(point_start[1], point_end[1]) - 5, 0)
    col_max = min(max(point_start[1], point_end[1]) + 5, image.shape[1])

    roi = processed_image[row_min:row_max, col_min:col_max]
    total_intensity = 0
    valid_pixel_count = 0

    for row in range(roi.shape[0]):
        for col in range(roi.shape[1]):
            if roi[row, col] > 150:
                valid_pixel_count += 1
                total_intensity += roi[row, col]

    adaptive_threshold = (total_intensity / valid_pixel_count) * intensity_threshold

    # 2a. apply thresholding
    binary_image = do_adaptive_thresholding(processed_image)
    thresholded_image = do_binary_thresholding(contrast_image, adaptive_threshold - 5)
    thresholded_image = thresholded_image.astype(np.uint8)
    thresholded_image = normal_opening(thresholded_image, 2)

    combined_binary = binary_image * (thresholded_image / 255)
    cropped_binary = crop_image(combined_binary,
                                max(row_min - 100, 0), min(row_max + 100, combined_binary.shape[0]),
                                max(col_min - 100, 0), min(col_max + 100, combined_binary.shape[1]))

    final_binary = normal_opening(cropped_binary, 2).astype(np.uint8)

    # 3. find connected components
    _, component_labels = cv.connectedComponents(final_binary)

    # remove background label
    label_counts = np.bincount(component_labels.flatten())
    label_counts[0] = 0

    roi_labels = np.bincount(component_labels[row_min:row_max, col_min:col_max].flatten())
    roi_labels[0] = 0

    # 4. find components close to user-line
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
                    selected_labels.add(component_labels[row, col])

    # if no targets found, extend search region
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
                        selected_labels.add(component_labels[row, col])

    # filter binary image for selected labels
    selected_labels = list(selected_labels)
    filtered_labels = np.isin(component_labels, selected_labels).astype(np.uint8)
    final_binary = final_binary * filtered_labels

    # smooth
    final_binary = normal_closing(final_binary, 2)

    # get best line
    line_info = select_best_line(final_binary, point_start, point_end)
    if line_info is None:
        return

    [[y1, x1, y2, x2]], line_derivative, hough_lines, hough_line = line_info
    refined_binary = opening(final_binary, smoothing_kernel, line_derivative)

    refined_start = np.array([x1, y1])
    refined_end = np.array([x2, y2])

    # calculate center and line length
    line_center = (refined_start + refined_end) // 2
    line_length = np.linalg.norm(refined_end - refined_start)

    # remove outlier for more cleaning
    for row in range(max(row_min - 105, 0), min(row_max + 105, refined_binary.shape[0])):
        for col in range(max(col_min - 105, 0), min(col_max + 105, refined_binary.shape[1])):
            if refined_binary[row, col] != 0:
                point = np.array([row, col])
                distance_to_line = abs(np.cross(refined_end - refined_start, point - refined_start)
                                       / np.linalg.norm(refined_end - refined_start))
                distance_to_center = np.linalg.norm(point - line_center)
                if distance_to_line > 5 or distance_to_center > (6 / 11) * line_length:
                    refined_binary[row, col] = 0

    return [refined_start, refined_end], refined_binary.astype(np.uint8), line_length
