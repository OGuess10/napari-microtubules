import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt


def select_best_line(image, target_pixel_start, target_pixel_end):
    """ choose the best line from detected lines using loss """
    target_pixel_start, target_pixel_end = np.array(target_pixel_start), np.array(target_pixel_end)
    target_line_length = np.linalg.norm(target_pixel_end - target_pixel_start)


    def otsu_threshold(image):
        """ do Otsu's method manually with smoothing """
        # smooth image to reduce noise
        smoothed_image = cv.GaussianBlur(image, (5, 5), 0)
        
        # flatten image and calculate histogram
        hist, bin_edges = np.histogram(smoothed_image.flatten(), bins=256, range=(0, 256))
        
        # get total number of pixels
        total_pixels = smoothed_image.size
        
        # calculate the cumulative sum of the histogram
        cum_sum = np.cumsum(hist)
        
        # calculate the cumulative mean
        cum_mean = np.cumsum(hist * np.arange(256)) / cum_sum
        
        # initialize variables for maximum variance and optimal threshold
        max_variance = 0
        optimal_threshold = 0
        
        for threshold in range(1, 256):
            # class 1 (below threshold)
            weight1 = cum_sum[threshold - 1] / total_pixels
            mean1 = cum_mean[threshold - 1]
            
            # class 2 (above threshold)
            weight2 = (cum_sum[-1] - cum_sum[threshold - 1]) / total_pixels
            mean2 = (cum_mean[-1] - cum_mean[threshold - 1]) / (cum_sum[-1] - cum_sum[threshold - 1])
            
            # calculate between-class variance
            variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
            
            # update the optimal threshold if the new variance is greater
            if variance_between > max_variance:
                max_variance = variance_between
                optimal_threshold = threshold
        
        return optimal_threshold


    # apply Otsu's thresholding
    otsu_threshold_value = otsu_threshold(image)
    _, binary_image = cv.threshold(image, otsu_threshold_value, 255, cv.THRESH_BINARY)

    # Prep images for visualization
    all_detected_lines_image = np.zeros_like(binary_image)
    best_detected_line_image = np.zeros_like(binary_image)


    def calculate_angle_between_points(start, end):
        """ calculate angle of the line formed by two points """
        delta_x, delta_y = end[0] - start[0], end[1] - start[1]
        angle = math.atan2(delta_y, delta_x)
        return angle if angle >= 0 else angle + 2 * math.pi


    def calculate_line_loss(x_start, x_end, y_start, y_end, pixel_start, pixel_end, line_start, line_end, target_angle, line_angle, line_length, distance_weight, rotation_weight):
        """ calculate loss for a line, using distance, length, and rotation."""
        # distance loss: measure how far pixels are from line endpoints
        distance_loss = min(
            distance_weight * (np.linalg.norm(line_start - pixel_start) ** 2 + np.linalg.norm(line_end - pixel_end) ** 2),
            distance_weight * (np.linalg.norm(line_start - pixel_end) ** 2 + np.linalg.norm(line_end - pixel_start) ** 2),
        )
        
        # length loss: how the length of the detected line compares to the expected length
        length_loss = line_length / np.hypot(x_start - x_end, y_start - y_end)
        
        # rotation loss: how much the angle differs from the expected angle
        rotation_loss = rotation_weight * abs(target_angle - line_angle)
        
        return distance_loss + length_loss + rotation_loss


    # Calculate expected angle of the target line
    target_line_angle = calculate_angle_between_points(target_pixel_start, target_pixel_end)

    # Detect lines using Hough Transform
    detected_lines = cv.HoughLinesP(binary_image, 1, np.pi / 180, threshold=50, maxLineGap=10)
    if detected_lines is None:
        return None

    # Initialize best line selection variables
    best_line, min_loss, best_line_slope = None, float('inf'), None

    def draw_line(matrix, x_start, y_start, x_end, y_end, inplace=False):
        """ draw a line on the matrix using DDA (Digital Differential Analyzer) algorithm """
        if not inplace:
            matrix = matrix.copy()

        # Calculate differences and steps
        delta_x = x_end - x_start
        delta_y = y_end - y_start
        num_steps = max(abs(delta_x), abs(delta_y))  # Number of points to plot

        # Calculate increment for each step
        x_increment = delta_x / num_steps
        y_increment = delta_y / num_steps

        # Draw the line
        x, y = x_start, y_start
        for _ in range(num_steps + 1):
            x_int, y_int = round(x), round(y)
            if 0 <= x_int < matrix.shape[0] and 0 <= y_int < matrix.shape[1]:  # Check boundaries
                matrix[x_int, y_int] = 1
            x += x_increment
            y += y_increment

        return matrix if not inplace else None
    
    # Iterate through detected lines and calculate loss for each
    for line in detected_lines:
        y_start, x_start, y_end, x_end = line[0]
        
        # Draw all lines for visualization
        draw_line(all_detected_lines_image, x_start, y_start, x_end, y_end, inplace=True)
        
        # Calculate loss for the current line
        line_start, line_end = np.array([x_start, y_start]), np.array([x_end, y_end])
        line_slope = (x_start - x_end) / (y_start - y_end + 0.001)
        line_angle = calculate_angle_between_points(line_start, line_end)
        
        total_loss = calculate_line_loss(
            x_start, x_end, y_start, y_end,
            target_pixel_start, target_pixel_end,
            line_start, line_end,
            line_angle, target_line_angle, 
            target_line_length, 0.005, 15
        )

        # Update best line if the loss is smaller
        if total_loss < min_loss:
            min_loss = total_loss
            best_line = line
            best_line_slope = line_slope
            best_detected_line_image.fill(0)  # Reset and draw ONLY the best line
            draw_line(best_detected_line_image, x_start, y_start, x_end, y_end, inplace=True)

    return best_line, best_line_slope, all_detected_lines_image, best_detected_line_image
