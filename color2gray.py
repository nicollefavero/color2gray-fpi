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

def colorToGray(imagePath):

    # edit here image name to be convert to grayscale
    #image_name = "images/mapIslandSmall.png"
    image_name = imagePath
    original_image = cv.imread(image_name)
    image_lab = cv.cvtColor(original_image, cv.COLOR_BGR2LAB)

    # image attributes
    height, width= image_lab.shape[:2]
    h = range(height)
    w = range(width)

    # defaults are alpha = 10, theta = 45, neighbourhood_size = image_size
    alpha = 20
    theta = 45
    neighbourhood_size = (width, height)

    hn = range(neighbourhood_size[1])
    wn = range(neighbourhood_size[0])
    N = width * height

    # convert the original image to normal grayscale and initialize output
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    final_image = np.zeros_like(original_image)

    # initialize target difference matrix
    #target_difference = np.zeros((height, width), dtype=float)
    target_difference = np.zeros(N, dtype=float)

    # Lumiance List, Channel A List, Channel B List
    LiList = np.zeros(N, dtype=float)
    AiList = np.zeros(N, dtype=float)
    BiList = np.zeros(N, dtype=float)

    # i pixel and j is its neighbour
    # go through each pixel i of the original image
    for yi in h:
      for xi in w:

        Li, Ai, Bi = image_lab[yi][xi]
        LiList[(yi * width + xi)] = Li
        AiList[(yi * width + xi)] = Ai
        BiList[(yi * width + xi)] = Bi

    # go through each pixel j of the neighbourhood size
    for yi in range(N):
      for xi in range(N):
        delta_L = LiList[yi] - LiList[xi]
        delta_A = AiList[yi] - AiList[xi]
        delta_B = BiList[yi] - BiList[xi]

        target_difference[yi] += calculateTargetDifference(alpha, theta, delta_L, delta_A, delta_B)        


    # solves the optimization using the target differences
    for i in range(1, N):
      LiList[i] = target_difference[i] - target_difference[i-1] + N * LiList[i-1]
      LiList[i] /= N

    for i in range(height):
      for j in range(width):
        final_image[i][j] = LiList[i * width + j]
    
    cv.imwrite("output.png", final_image)
