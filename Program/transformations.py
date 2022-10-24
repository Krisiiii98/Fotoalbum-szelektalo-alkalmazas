import os
import cv2
import math
import numpy as np
from PIL import Image

images = []
image_names = []
i = 0

#Képek betöltése a kiválasztott mappából
def upload_images():
    folder_selected = filedialog.askdirectory()
    for filepath in os.listdir(folder_selected):
        images.append(cv2.imread('images/{0}'.format(filepath),1))
        image_names.append(filepath)

#Képek átméretezése a képarány megtartásával (ratio)
def image_resize(image):
    np_img = np.array(image)
    ratio = 1080/np_img.shape[0]
    M = int(np_img.shape[1]*ratio)
    if M >= 1920:
        M = 1920
    else:
        M = M
    dim = (M,1080)
    stretch_near = cv2.resize(np_img, dim,interpolation = cv2.INTER_AREA)
    pil_image = Image.fromarray(stretch_near)
    return pil_image

#Hisztogram kiegyenlítés alul- vagy túlexponált képek javítására
def Histogram_equalization(image):
    global i
    img_array = np.array(image)
    length = len(img_array.shape)
    if length == 2:
        typ = "gray"
        img_hsv = img.copy()
        img_hsv_array = np.array(img_hsv)
        i_level, n_pixels  = np.unique(img_hsv_array, return_counts = True)
    else:
        typ = "RGB"
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        img_hsv_array = np.array(img_hsv)
        i_level, n_pixels = np.unique(img_hsv_array[:,:,2], return_counts = True) 

    M = img_hsv_array.shape[0]
    N = img_hsv_array.shape[1] 
    p_pixel = n_pixels/(M*N) 
    cump_r = np.cumsum(p_pixel)
    s_k = 255*cump_r
    s_k = np.rint(s_k)

    Iter = dict(zip(i_level,s_k))
    def replace(r):
        return Iter[r]
    replace_v = np.vectorize(replace)

    if typ == "RGB":
        img_hsv_array[:,:,2] = replace_v(img_hsv_array[:,:,2])
        final_image_array = cv2.cvtColor(img_hsv_array, cv2.COLOR_HSV2RGB)
    else:
        img_hsv_array = replace_v(img_hsv_array)
        final_image_array = img_hsv_array
    pil_img = Image.fromarray(final_image_array)
    image = np.array(pil_img)

    #Javított képek mentése
    cv2.imwrite(image_names[i], image)
    i += 1