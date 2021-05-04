import numpy as np
from cv2 import cv2 as cv
import matplotlib as plt
import math

def crunch(alpha, x):
  # This function allows us to compress large-amplitude values
  # into the valid range, while leaving small yet significant
  # variations around zero almost untouched
  return alpha * math.tanh(x/alpha)


def calculateTargetDifference(alpha, theta, delta_lum, delta_A, delta_B):
  # normalized vector defined by theta (relative to the delta_A 
  delta_crom_norm = math.sqrt(math.pow(delta_A, 2) + math.pow(delta_B, 2))
  dot_product_AB_theta = delta_A * math.cos(theta) + delta_B * math.sin(theta)
  
  if np.abs(delta_lum) > crunch(alpha, delta_crom_norm):
    return delta_lum

  elif dot_product_AB_theta >= 0:
    return crunch(alpha, delta_crom_norm)

  else:
    return crunch(alpha, -delta_crom_norm)

if __name__ == "__main__":

    # edit here image name to be convert to grayscale
    image_name = "original_color_image.png"
    original_image = cv.imread(image_name)
    image_lab = cv.cvtColor(original_image, cv.COLOR_BGR2LAB)

    # image attributes
    height, width= image_lab.shape[:2]
    h = range(height)
    w = range(width)

    # defaults are alpha = 10, theta = 45, neighbourhood_size = image_size
    alpha = 10
    theta = 45
    neighbourhood_size = (width, height)

    hn = range(neighbourhood_size[1])
    wn = range(neighbourhood_size[0])

    # convert the original image to normal grayscale and initialize output
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    final_image = np.zeros_like(original_image)

    # initialize target difference matrix
    target_difference = np.zeros((height, width), dtype=float)

    # Lumiance List, Channel A List, Channel B List
    LiList = []
    AiList = []
    BiList = []

    # i pixel and j is its neighbour
    # go through each pixel i of the original image
    for yi in h:
      for xi in w:

        Li, Ai, Bi = image_lab[yi][xi]
        LiList.append(Li)
        AiList.append(Ai)
        BiList.append(Bi)

    # go through each pixel j of the neighbourhood size
    for yj in hn:
      for xj in wn:
        Lj, Aj, Bj = image_lab[yj][xj]

        delta_L = int(LiList[yj]) - Lj   # black - white
        delta_A = int(AiList[yj]) - Aj   # blue - yellow
        delta_B = int(BiList[yj]) - Bj   # green - red

        # a pseudocode to explain how target difference works: target_difference[yi][xi][yj][xj] = int()
        # here, a simplified version because the neighbours are the same for a given pixel
        target_difference[yj][xj] = calculateTargetDifference(alpha, theta, delta_L, delta_A, delta_B)


    # solves the optimization using the target differences
    for yi in hn:
      for xi in wn:
        if yi - 1 < 0:
          target_diff_dim = 0
        else:
          target_diff_dim = np.sum(target_difference[yi]) - np.sum(target_difference[yi - 1]) + width * height * LiList[yi * width + xi - 1]
          target_diff_dim /= (width * height)
        
        final_image[yi][xi] = target_diff_dim

    error = 0

    # reduces the brightness error by calculating the average distance from the gray image
    for yi in h:
      for xi in w:
        error += (final_image[yi][xi] - gray_image[yi][xi])[0]

    error /= (width * height)
    error = np.uint8(error)

    for yi in h:
      for xi in w:
        final_image[yi][xi] -= error
    
    cv.imwrite("output.png", final_image)
