import cv2 as cv
import math
import numpy as np


def get_threshold_for_line_length(length):
    """ determine threshold and gap values based on line length. """
    if length > 130:
        return 50, 20
    elif length > 80:
        return 30, 15
    elif length > 50:
        return 20, 10
    return 15, 5


def calculate_angle_between_points(p1, p2):
    """ calculate angle of line formed by two points """
    slope = (p1[0] - p2[0]) / (p1[1] - p2[1] + 0.001)
    return math.atan(slope) + (math.pi if slope < 0 else 0)


def calculate_line_loss(x1, x2, y1, y2, pixel1, pixel2, p1, p2, angle, calculated_angle, length, w1, w2):
    """ calculate loss for a given line, considering distance, length, and rotation """
    # distance loss: measure how far pixels are from line end
    distance_loss = min(
        w1 * (np.linalg.norm(p1 - pixel1) ** 2 + np.linalg.norm(p2 - pixel2) ** 2),
        w1 * (np.linalg.norm(p1 - pixel2) ** 2 + np.linalg.norm(p2 - pixel1) ** 2),
    )
    
    # length loss: how length of the detected line compares to expected length
    length_loss = length / np.hypot(x1 - x2, y1 - y2)
    
    # rotation loss: how much angle differs from expected angle
    rotation_loss = w2 * abs(angle - calculated_angle)
    
    return distance_loss + length_loss + rotation_loss


def select_best_line(image, pix1, pix2):
    """ choose best line from detected lines using loss calculation """
    pix1, pix2 = np.array(pix1), np.array(pix2)
    length = np.linalg.norm(pix2 - pix1)
    
    # get threshold and gap based on line length
    thresh, gap = get_threshold_for_line_length(length)

    # set weight factors for distance and rotation losses
    w1, w2 = 0.005, 15  

    # prep images for visualization
    all_lines_image = np.zeros_like(image)
    best_line_image = np.zeros_like(image)

    # calculate expected angle of target line
    calculated_angle = calculate_angle_between_points(pix1, pix2)

    # detect lines using Hough Transform
    lines = cv.HoughLinesP(image, 1, np.pi / 180, threshold=thresh, maxLineGap=gap)
    if lines is None:
        return None

    # initialize best line selection variables
    best_line, min_loss, best_slope = None, float('inf'), None
    
    # iterate through detected lines and calculate loss for each
    for line in lines:
        y1, x1, y2, x2 = line[0]
        
        # draw all lines for visualization
        draw_line(all_lines_image, x1, y1, x2, y2, inplace=True)
        
        # calculate loss for current line
        p1, p2 = np.array([x1, y1]), np.array([x2, y2])
        d = (x1 - x2) / (y1 - y2 + 0.001)
        angle = calculate_angle_between_points(p1, p2)
        
        total_loss = calculate_line_loss(x1, x2, y1, y2, pix1, pix2, p1, p2, angle, calculated_angle, length, w1, w2)

        # update best line if loss is smaller
        if total_loss < min_loss:
            min_loss = total_loss
            best_line = line
            best_slope = d
            best_line_image.fill(0)  # reset and draw ONLY best line
            draw_line(best_line_image, x1, y1, x2, y2, inplace=True)

    return best_line, best_slope, all_lines_image, best_line_image

def draw_line(mat, x0, y0, x1, y1, inplace=False):
    """ draw a line on the matrix """
    if not inplace:
        mat = mat.copy()

    # handle case where start and end points are the same
    if (x0, y0) == (x1, y1):
        mat[x0, y0] = 2
        return mat if not inplace else None

    # make sure left-to-right case is accounted for
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0

    # draw line with vectorization
    x = np.arange(x0, x1)
    y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(int)
    mat[x0, y0], mat[x1, y1] = 2, 2  # mark start and end points
    mat[x, y] = 1  # mark the line pixels

    return mat if not inplace else None
