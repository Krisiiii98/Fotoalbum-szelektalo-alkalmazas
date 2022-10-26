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

#Képek átméretezése a képarány megtartásával (ratio)
def image_resize(image):
    np_img = np.array(image)
    ratio = 1080/(np.array(np_img.shape[0]))
    M = int(np_img.shape[1]*ratio)
    if M >= 1920:
        M = 1920
    else:
        M = M
    dim = (M,1080)
    stretch_near = cv2.resize(np_img, dim,interpolation = cv2.INTER_AREA)
    return Image.fromarray(stretch_near)

#Hisztogram kiegyenlítés alul- vagy túlexponált képek javítására
def Histogram_equalization(image):
    img_array = np.array(image)
    length = len(img_array.shape)
    if length == 2:
        typ = "gray"
        img_hsv = image.copy()
        img_hsv_array = np.array(img_hsv)
        i_level, n_pixels  = np.unique(img_hsv_array, return_counts = True)
    else:
        typ = "RGB"
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        img_hsv_array = np.array(img_hsv)
        i_level, n_pixels = np.unique(img_hsv_array[:,:,2], return_counts = True) #megadja a képpontok számát(n_pixels) az intenzitási szintben(i_level)

    M = img_hsv_array.shape[0]
    N = img_hsv_array.shape[1] #MxN a pixelek száma
    p_pixel = n_pixels/(M*N) #az adott intenzitású pixel előfordulásának valószínűsége(p_pixel)
    cump_p = np.cumsum(p_pixel) #halmozott összeg
    s_k = 255*cump_p #transzformáció az intenzitások kiegyenlítésére
    s_k = np.rint(s_k) #kerekítés közeli egész számokra

    Iter = dict(zip(i_level,s_k)) #mapping i_level és s_k
    def replace(r):
        return Iter[r] #az eredeti i_level-t s_k helyettesítjük
    replace_v = np.vectorize(replace) #Helyettesítés vektorizálással

    if typ == "RGB":
        img_hsv_array[:,:,2] = replace_v(img_hsv_array[:,:,2])
        final_image_array = cv2.cvtColor(img_hsv_array, cv2.COLOR_HSV2RGB) #hsv kép konvertálása vissza RGB
    else:
        img_hsv_array = replace_v(img_hsv_array)
        final_image_array = img_hsv_array
    pil_img = Image.fromarray(final_image_array)
    image = np.array(pil_img)
    return image