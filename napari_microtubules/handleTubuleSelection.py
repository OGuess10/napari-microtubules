import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

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


def calculate_angle_between_points(p1, p2):
    """Calculate the angle of the line formed by two points."""
    slope = (p1[0] - p2[0]) / (p1[1] - p2[1] + 0.001)
    return math.atan(slope) + (math.pi if slope < 0 else 0)


def calculate_line_loss(x1, x2, y1, y2, pixel1, pixel2, p1, p2, angle, calculated_angle, length, w1, w2):
    """ calculate loss for a given line, using distance, length, and rotation """
    # distance loss: measure how far pixels are from line ends
    distance_loss = min(
        w1 * (np.linalg.norm(p1 - pixel1) ** 2 + np.linalg.norm(p2 - pixel2) ** 2),
        w1 * (np.linalg.norm(p1 - pixel2) ** 2 + np.linalg.norm(p2 - pixel1) ** 2),
    )
    
    # length loss: how length of the detected line compares to the expected length
    length_loss = length / np.hypot(x1 - x2, y1 - y2)
    
    # rotation loss: how much the angle differs from the expected angle
    rotation_loss = w2 * abs(angle - calculated_angle)
    
    return distance_loss + length_loss + rotation_loss


def select_best_line(image, pix1, pix2):
    """ choose the best line from detected lines using loss calculation."""
    pix1, pix2 = np.array(pix1), np.array(pix2)
    length = np.linalg.norm(pix2 - pix1)

    # Apply Otsu's thresholding manually
    otsu_threshold_value = otsu_threshold(image)
    _, binary_image = cv.threshold(image, otsu_threshold_value, 255, cv.THRESH_BINARY)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Binarized Image (Otsu)")
    # plt.imshow(smoothed_image, cmap='gray')
    # plt.axis('off')

    # Set weight factors for distance and rotation losses
    w1, w2 = 0.005, 15  

    # Prep images for visualization
    all_lines_image = np.zeros_like(binary_image)
    best_line_image = np.zeros_like(binary_image)

    # Calculate expected angle of the target line
    calculated_angle = calculate_angle_between_points(pix1, pix2)

    # Detect lines using Hough Transform
    lines = cv.HoughLinesP(binary_image, 1, np.pi / 180, threshold=50, maxLineGap=10)
    if lines is None:
        return None

    # Initialize best line selection variables
    best_line, min_loss, best_slope = None, float('inf'), None


    def draw_line(mat, x0, y0, x1, y1, inplace=False):
        """Draw a line on the matrix."""
        if not inplace:
            mat = mat.copy()

        # Handle case where start and end points are the same
        if (x0, y0) == (x1, y1):
            mat[x0, y0] = 2
            return mat if not inplace else None

        # Make sure left-to-right case is accounted for
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0

        # Draw line with vectorization
        x = np.arange(x0, x1)
        y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(int)
        mat[x0, y0], mat[x1, y1] = 2, 2  # Mark start and end points
        mat[x, y] = 1  # Mark the line pixels

        return mat if not inplace else None
    
    
    # Iterate through detected lines and calculate loss for each
    for line in lines:
        y1, x1, y2, x2 = line[0]
        
        # Draw all lines for visualization
        draw_line(all_lines_image, x1, y1, x2, y2, inplace=True)
        
        # Calculate loss for the current line
        p1, p2 = np.array([x1, y1]), np.array([x2, y2])
        d = (x1 - x2) / (y1 - y2 + 0.001)
        angle = calculate_angle_between_points(p1, p2)
        
        total_loss = calculate_line_loss(x1, x2, y1, y2, pix1, pix2, p1, p2, angle, calculated_angle, length, w1, w2)

        # Update best line if the loss is smaller
        if total_loss < min_loss:
            min_loss = total_loss
            best_line = line
            best_slope = d
            best_line_image.fill(0)  # Reset and draw ONLY the best line
            draw_line(best_line_image, x1, y1, x2, y2, inplace=True)

    # plt.subplot(1, 2, 2)
    # plt.title("Hough Lines")
    # plt.imshow(cv.cvtColor(best_line_image, cv.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    return best_line, best_slope, all_lines_image, best_line_image



# import cv2 as cv
# import numpy as np
# import math

# def otsu_threshold(image):
#     """ do Otsu's method manually with smoothing """
#     # smooth image to reduce noise
#     smoothed_image = cv.GaussianBlur(image, (5, 5), 0)
    
#     # flatten image and calculate histogram
#     hist, bin_edges = np.histogram(smoothed_image.flatten(), bins=256, range=(0, 256))
    
#     # get total number of pixels
#     total_pixels = smoothed_image.size
    
#     # calculate the cumulative sum of the histogram
#     cum_sum = np.cumsum(hist)
    
#     # calculate the cumulative mean
#     cum_mean = np.cumsum(hist * np.arange(256)) / cum_sum
    
#     # initialize variables for maximum variance and optimal threshold
#     max_variance = 0
#     optimal_threshold = 0
    
#     for threshold in range(1, 256):
#         # class 1 (below threshold)
#         weight1 = cum_sum[threshold - 1] / total_pixels
#         mean1 = cum_mean[threshold - 1]
        
#         # class 2 (above threshold)
#         weight2 = (cum_sum[-1] - cum_sum[threshold - 1]) / total_pixels
#         mean2 = (cum_mean[-1] - cum_mean[threshold - 1]) / (cum_sum[-1] - cum_sum[threshold - 1])
        
#         # calculate between-class variance
#         variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
        
#         # update the optimal threshold if the new variance is greater
#         if variance_between > max_variance:
#             max_variance = variance_between
#             optimal_threshold = threshold
    
#     return optimal_threshold


# def calculate_angle_between_points(p1, p2):
#     """Calculate the angle of the line formed by two points."""
#     slope = (p1[0] - p2[0]) / (p1[1] - p2[1] + 0.001)
#     return math.atan(slope) + (math.pi if slope < 0 else 0)


# def calculate_line_loss(x1, x2, y1, y2, pixel1, pixel2, p1, p2, angle, calculated_angle, length, w1, w2):
#     """Calculate loss for a given line, considering distance, length, and rotation."""
#     # Distance loss: measure how far pixels are from line ends
#     distance_loss = min(
#         w1 * (np.linalg.norm(p1 - pixel1) ** 2 + np.linalg.norm(p2 - pixel2) ** 2),
#         w1 * (np.linalg.norm(p1 - pixel2) ** 2 + np.linalg.norm(p2 - pixel1) ** 2),
#     )
    
#     # Length loss: how length of the detected line compares to the expected length
#     line_length = max(np.hypot(x1 - x2, y1 - y2), 1e-6)
#     length_loss = length / line_length
    
#     # Rotation loss: how much the angle differs from the expected angle
#     rotation_loss = w2 * abs(angle - calculated_angle)
    
#     return distance_loss + length_loss + rotation_loss


# def select_best_line(image, pix1, pix2, threshold):
#     """Choose the best line from detected lines using loss calculation."""
#     pix1, pix2 = np.array(pix1), np.array(pix2)
#     length = np.linalg.norm(pix2 - pix1)

#     # Apply Otsu's thresholding manually
#     otsu_threshold_value = otsu_threshold(image)
#     _, binary_image = cv.threshold(image, otsu_threshold_value, 255, cv.THRESH_BINARY)

#     # Set gap for the Hough Transform
#     max_gap = 10 

#     # Set weight factors for distance and rotation losses
#     w1, w2 = 0.005, 15  

#     # Prep images for visualization
#     all_lines_image = np.zeros_like(binary_image)
#     best_line_image = np.zeros_like(binary_image)

#     # Calculate expected angle of the target line
#     calculated_angle = calculate_angle_between_points(pix1, pix2)

#     # Detect lines using Hough Transform
#     lines = cv.HoughLinesP(binary_image, 1, np.pi / 180, threshold, maxLineGap=max_gap)
#     if lines is None:
#         return None

#     # Initialize best line selection variables
#     best_line, min_loss, best_slope = None, float('inf'), None
    
#     # Iterate through detected lines and calculate loss for each
#     for line in lines:
#         y1, x1, y2, x2 = line[0]
        
#         # Draw all lines for visualization
#         draw_line(all_lines_image, x1, y1, x2, y2, inplace=True)
        
#         # Calculate loss for the current line
#         p1, p2 = np.array([x1, y1]), np.array([x2, y2])
#         d = (x1 - x2) / (y1 - y2 + 0.001)
#         angle = calculate_angle_between_points(p1, p2)
        
#         total_loss = calculate_line_loss(x1, x2, y1, y2, pix1, pix2, p1, p2, angle, calculated_angle, length, w1, w2)

#         # Update best line if the loss is smaller
#         if total_loss < min_loss:
#             min_loss = total_loss
#             best_line = line
#             best_slope = d
#             best_line_image.fill(0)  # Reset and draw ONLY the best line
#             draw_line(best_line_image, x1, y1, x2, y2, inplace=True)

#     return best_line, best_slope, all_lines_image, best_line_image


# def draw_line(mat, x0, y0, x1, y1, inplace=False):
#     """Draw a line on the matrix."""
#     if not inplace:
#         mat = mat.copy()

#     # Handle case where start and end points are the same
#     if (x0, y0) == (x1, y1):
#         mat[x0, y0] = 2
#         return mat if not inplace else None

#     # Make sure left-to-right case is accounted for
#     if x0 > x1:
#         x0, y0, x1, y1 = x1, y1, x0, y0

#     # Draw line with vectorization
#     x = np.arange(x0, x1)
#     if x1 == x0:
#         y = np.full_like(x, y0, dtype=int)
#     else:
#         y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(int)
#     mat[x0, y0], mat[x1, y1] = 2, 2  # Mark start and end points
#     mat[x, y] = 1  # Mark the line pixels

#     return mat if not inplace else None

