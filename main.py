import numpy as np
from cv2 import cv2 as cv
import matplotlib as plt
import math

def crunch(alpha, x):
  # This function allows us to compress large-amplitude values
  # into the valid range, while leaving small yet significant
  # variations around zero almost untouched

  return alpha * math.tanh(x/alpha)

def euclidianNorm(vector):
  # Transform a 2D vector into a single dimension value
  return np.linalg.norm(vector)

def calculateTargetDifference(alpha, theta, delta_lum, delta_crom):
  # normalized vector defined by theta (relative to the delta_A 
  # axis (???))
  norm_vector_theta = [math.cos(theta), math.sin(theta)]
  delta_crom_norm = euclidianNorm(delta_crom)
  
  if np.abs(delta_lum) > crunch(alpha, delta_crom_norm):
    return delta_lum

  elif np.dot(delta_crom, norm_vector_theta) >= 0:
    return crunch(alpha, delta_crom_norm)

  else:
    return crunch(alpha, -delta_crom_norm)

if __name__ == "__main__":

    # edit here image name to be convert to grayscale
    image_name = "original_reduced.png"
    original_image = cv.imread(image_name)
    image_lab = cv.cvtColor(original_image, cv.COLOR_BGR2LAB)

    # image attributes
    height, width= image_lab.shape[:2]
    h = range(height)
    w = range(width)

    # edit parameters here
    # defaults are alpha = 10, theta = 45, neighbourhood_size = image_size
    alpha = 10
    theta = 45
    neighbourhood_size = (width, height)

    hn = range(neighbourhood_size[1])
    wn = range(neighbourhood_size[0])

    # convert the original image to normal grayscale and initialize output
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    final_image = np.zeros_like(original_image)

    # i pixel and j is its neighbour
    # go through each pixel i of the original image
    for yi in h:
        for xi in w:

            Li, Ai, Bi = image_lab[yi][xi]

            # initialize target difference matrix
            target_difference = np.zeros((height, width), dtype=float)

            # sum of optimization least squares
            sum = 0

            # go through each pixel j of the neighbourhood size
            for yj in hn:
                for xj in wn:
                    
                    Lj, Aj, Bj = image_lab[yj][xj]

                    delta_L = int(Li) - Lj   # black - white
                    delta_A = int(Ai) - Aj   # blue - yellow
                    delta_B = int(Bi) - Bj   # green - red

                    # delta_C is a 2D vector related to the chromatic difference
                    delta_C = [delta_A, delta_B]

                    # a pseudocode to explain how target difference works: target_difference[yi][xi][yj][xj] = int()
                    # here, a simplified version because the neighbours are the same for a given pixel
                    target_difference[yj][xj] = calculateTargetDifference(alpha, theta, delta_L, delta_C)


            # transforms this target_difference[][][yj][xj] into a integer for each pixel [yi][xi] through a sum
            # check how to compute least squares of function below, see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
            target_diff_dim = ((gray_image[yi][xi] - gray_image[yj][xj]) - target_difference[yj][xj]) ** 2
                  
            final_image[yi][xi] = np.uint8(np.clip(sum, 0, 255))
        
    cv.imwrite("output.png", final_image)
