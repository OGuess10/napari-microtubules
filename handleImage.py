import numpy as np
from morphOperations import *
from handleTubuleSelection import *


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


def detectLine(img, line, k, threshold):
    """ track target microtubule based on user line input """
    pix1 = [round(line[0][1]), round(line[0][2])]
    pix2 = [round(line[1][1]), round(line[1][2])]

    # 1. blur and adjust global contrast
    img2, img = denoiseImage(img)

    # 2. find threshold value to get binary image
    temp11 = max(min(pix1[0], pix2[0]) - 5, 0)
    temp12 = min(max(pix1[0], pix2[0]) + 5, img.shape[0])
    temp21 = max(min(pix1[1], pix2[1]) - 5, 0)
    temp22 = min(max(pix1[1], pix2[1]) + 5, img.shape[1])
    total = 0
    count_nonzero = 0
    thresholdmatrix = img[temp11:temp12, temp21:temp22]
    for i in range(thresholdmatrix.shape[0]):
        for j in range(thresholdmatrix.shape[1]):
            if thresholdmatrix[i, j] > 150:
                count_nonzero += 1
                total += thresholdmatrix[i, j]
    thres = total / count_nonzero * threshold
    
    # 2a. do thresholding
    bin_img = doAdaptiveThresholding(img)
    img2 = doBinaryThresholding(img2, thres - 5)
    img2 = img2.astype(np.uint8)
    img2 = normal_opening(img2, 2)

    bin_img = bin_img * (img2 / 255)

    bin_img = cropImage(bin_img, max(temp11 - 100, 0), min(temp12 + 100, bin_img.shape[0]),
                       max(temp21 - 100, 0), min(temp22 + 100, bin_img.shape[1]))

    bin_img = normal_opening(bin_img, 2)
    bin_img = bin_img.astype(np.uint8)


    _, label = cv.connectedComponents(bin_img)

    counts_all = np.bincount(np.ndarray.flatten(label))
    counts_all[0] = 0

    counts = np.bincount(np.ndarray.flatten(label[temp11: temp12, temp21:temp22]))
    counts[0] = 0

    # 4. find all connected components in incline rectangle area formed by input
    targets = set()
    for i in range(max(temp11 + 3, 0), min(temp12 - 3, bin_img.shape[0])):
        for j in range(max(temp21 + 3, 0), min(temp22 - 3, bin_img.shape[1])):
            if bin_img[i, j] != 0:
                p3 = np.array([i, j])
                d = abs(np.cross(np.array(pix2) - np.array(pix1), p3 - np.array(pix1)) / np.linalg.norm(
                    np.array(pix2) - np.array(pix1)))
                if d < 5:
                    targets.add(label[i, j])
    
    targets = list(targets)
    if len(targets) == 0:
        targets = set()
        for i in range(max(temp11 - 5, 0), min(temp12 + 5, bin_img.shape[0])):
            for j in range(max(temp21 - 5, 0), min(temp22 + 5, bin_img.shape[1])):
                if bin_img[i, j] != 0:
                    p3 = np.array([i, j])
                    d = abs(np.cross(np.array(pix2) - np.array(pix1), p3 - np.array(pix1)) / np.linalg.norm(
                        np.array(pix2) - np.array(pix1)))
                    if d < 15:
                        targets.add(label[i, j])
    
    targets = list(targets)
    label = np.isin(label, targets).astype(np.uint8)
    
    # 4a. extract all pixels of target labels
    bin_img = bin_img * label

    # 4c. smoothing
    bin_img = normal_closing(bin_img, 2)

    # 5. get all lines
    ret = select_line(bin_img, pix1, pix2)
    if ret is None:
        return
    [[y1, x1, y2, x2]], derivative, hglines, hgline = ret
    bin_img = opening(bin_img, k, derivative)

    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    
    # center
    pix3 = (np.array(p1) + np.array(p2)) // 2

    # 5a. statistical analysis
    l = np.linalg.norm(p2 - p1)

    # 6. delete remote points to line
    for i in range(max(temp11 - 105, 0), min(temp12 + 105, bin_img.shape[0])):
        for j in range(max(temp21 - 105, 0), min(temp22 + 105, bin_img.shape[1])):
            if bin_img[i, j] != 0:
                p3 = np.array([i, j])
                d = abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))
                d2 = np.linalg.norm(p3 - pix3)
                if d > 5 or d2 > 6 / 11 * l:
                    bin_img[i, j] = 0
    bin_img = bin_img.astype(np.uint8)
    end_points = [p1, p2]
    return end_points, bin_img, l
