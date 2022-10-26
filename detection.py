import tkinter as tk
from tkinter import filedialog
from tkinter import *
import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image 
from transformations import *
from detection import *

def estimate_blur(image: np.array, threshold: int = 100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    if(score < threshold):
        blurry = True
    else:
        blurry = False
    return blurry

def estimate_exposure(image: np.array, exposure_thresholds: int = [0, 51, 102, 153, 204, 256]):
    hist_gray = cv2.calcHist([image], [0], None, [256], [0, 256])

    max_pixel_brightness = 0
    max_pixel_brightness_place = 0

    for x in range(0, 256):
        if  max_pixel_brightness < hist_gray[x]:
            max_pixel_brightness = hist_gray[x]
            max_pixel_brightness_place = x

    if max_pixel_brightness_place >= 0 and max_pixel_brightness_place <= 51:
        print('Image is Very Underexposed')
        return True
    elif max_pixel_brightness_place > 51 and max_pixel_brightness_place <= 102:
        print('Image is Underexposed')
        return True
    elif max_pixel_brightness_place > 102 and max_pixel_brightness_place <= 153:  
        print('Image is Correct')
        return False
    elif max_pixel_brightness_place > 153 and max_pixel_brightness_place <= 204:
        print('Image is Overexposed') 
        return True  
    elif max_pixel_brightness_place > 204  and max_pixel_brightness_place <= 256:
        print('Image is Very Overexposed')
        return True

