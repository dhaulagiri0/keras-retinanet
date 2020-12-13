import cv2
import os
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

def otsu(image, is_normalized=True, is_reduce_noise=False):

    # Apply GaussianBlur to reduce image noise if it is required
    if is_reduce_noise:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    return threshold


def preprocessImage(name, imgPathString, savePath = None):
    """
    binarise image and applies morphological dilation and opening
    overwrites image unless given an alternative path
    if providing alternative path, makes sure the folder exists
    """

    img = cv2.imread(imgPathString, 0)

    # find threshold value with otsu's method
    thresh = otsu(img)

    # binarise
    _, binary = cv2.threshold(img , thresh, 255, cv2.THRESH_BINARY_INV)

    # morphological dilation
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations = 1)

    # morphological opening 
    # if iterations is set to be more than one the image dies
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel, iterations=1)

    # save
    if savePath != None:
        print(f"{savePath}\{name}")
        cv2.imwrite(f"{savePath}\{name}", opening)
    else:
        cv2.imwrite(imgPathString, opening)

if __name__ == "__main__":
    import argparse

         # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate segmentation masks of arteries')
    parser.add_argument('--data_path', required=True,
                        metavar="/path/to/images/",
                        help="Directory folder for images")
    parser.add_argument('--save_path', required=False,
                        metavar="/path/to/save folder",
                        help="Path to whichever folder the processed images will be saved")
    
    args = parser.parse_args()
    pathString = args.data_path
    savePath = args.save_path

    dataPath = Path(pathString)
    for img in dataPath.iterdir():
        preprocessImage(img.name, str(img.absolute()), savePath)
