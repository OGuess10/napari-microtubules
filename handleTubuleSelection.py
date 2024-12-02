import cv2 as cv
import math
import numpy as np


def _get_threshold(length):
    """ determine threshold and gap values based on line length """
    if length > 130:
        return 50, 20
    elif length > 80:
        return 30, 15
    elif length > 50:
        return 20, 10
    return 15, 5


def _calculate_angle(p1, p2):
    """ get angle of a line given two points for comparison """
    slope = (p1[0] - p2[0]) / (p1[1] - p2[1] + 0.001)
    return math.atan(slope) + (math.pi if slope < 0 else 0)


def _calculate_loss(x1, x2, y1, y2, pixel1, pixel2, p1, p2, angle, calculated_angle, length, w1, w2):
    """ calculate loss for a line """
    distance_loss = min(
        w1 * (np.linalg.norm(p1 - pixel1)**2 + np.linalg.norm(p2 - pixel2)**2),
        w1 * (np.linalg.norm(p1 - pixel2)**2 + np.linalg.norm(p2 - pixel1)**2),
    )
    # length_loss = length / np.linalg.norm(p2 - p1)
    length_loss = length / np.hypot(x1 - x2, y1 - y2)
    rotation_loss = w2 * abs(angle - calculated_angle)
    return distance_loss + length_loss + rotation_loss


def select_line(image, pix1, pix2):
    """ selects target line from detected lines """
    pix1, pix2 = np.array(pix1), np.array(pix2)
    length = np.linalg.norm(pix2 - pix1)
    
    thresh, gap = _get_threshold(length)

    w1, w2 = 0.005, 15 # weight factors

    all_lines_image = np.zeros_like(image)
    best_line_image = np.zeros_like(image)

    calculated_angle = _calculate_angle(pix1, pix2)

    lines = cv.HoughLinesP(image, 1, np.pi / 180, threshold=thresh, maxLineGap=gap)
    if lines is None:
        return None

    # get best line based on loss calc
    best_line, min_loss, best_slope = None, float('inf'), None
    
    for line in lines:
        y1, x1, y2, x2 = line[0]
        
        # draw all lines
        _draw_line(all_lines_image, x1, y1, x2, y2, inplace=True)
        
        # calculate loss for line
        p1, p2 = np.array([x1, y1]), np.array([x2, y2])
        d = (x1 - x2) / (y1 - y2 + 0.001)
        angle = _calculate_angle(p1, p2)
        # angle = math.atan(d) + (math.pi if d < 0 else 0)
        
        total_loss = _calculate_loss(x1, x2, y1, y2, pix1, pix2, p1, p2, angle, calculated_angle, length, w1, w2)

        # update best line
        if total_loss < min_loss:
            min_loss = total_loss
            best_line = line
            best_slope = d
            best_line_image.fill(0)  # reset and draw ONLY best line
            _draw_line(best_line_image, x1, y1, x2, y2, inplace=True)

    return best_line, best_slope, all_lines_image, best_line_image


def _draw_line(mat, x0, y0, x1, y1, inplace=False):
    """ draws a line on a matrix using Bresenham's algorithm """
    if not inplace:
        mat = mat.copy()

    # h, w = mat.shape
    # if not (0 <= x0 < h and 0 <= x1 < h and 0 <= y0 < w and 0 <= y1 < w):
    #     raise ValueError("invalid coordinates")

    if (x0, y0) == (x1, y1):
        mat[x0, y0] = 2
        return mat if not inplace else None

    # transpose for steep lines
    transpose = abs(x1 - x0) < abs(y1 - y0)
    if transpose:
        mat = mat.T
        x0, y0, x1, y1 = y0, x0, y1, x1

    # make sure left-to-right direction works when selecting microtubule
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0

    # draw line with interpolation
    x = np.arange(x0 + 1, x1)
    y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(int)
    mat[x0, y0], mat[x1, y1] = 2, 2
    mat[x, y] = 1

    return mat if not inplace else None
