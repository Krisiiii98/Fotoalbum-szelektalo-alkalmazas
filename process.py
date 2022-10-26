import tkinter as tk
from tkinter import filedialog
from tkinter import *
import cv2
import os
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image 
from transformations import *
from detection import *

i = 0
images = []
image_names = []

#Képek betöltése a kiválasztott mappából
def upload_images():
    folder_selected = filedialog.askdirectory()
    for filepath in os.listdir(folder_selected):
        images.append(cv2.imread('images\\{0}'.format(filepath),1))
        image_names.append(filepath)

#Felület létrehozása
gui = tk.Tk()
gui.geometry('400x200')
gui.title('Fotóalbum szelektáló')
gui.configure(background='#CDCDCD')

heading = Label(gui, text="Fotóalbum szelektáló", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack(side=TOP)

upload = Button(gui,text="Mappa kiválasztása", command=upload_images(), padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=TOP, pady=50)

label=Label(gui,background='#CDCDCD', font=('arial', 15, 'bold'))
label.pack(side=BOTTOM, expand=True)

#Képek átméretezése 

for image in images:
    images[i] = np.array(image_resize(image))
    i+=1

#Képek vizsgálata (képalkotási hibák detektálása)
gui.destroy()
i=0
for image in images:
    #Elmosódások felismerése
    blurry = estimate_blur(image, threshold= 100)
    if(blurry == True):
        print('Image is Blurred')
    else:
        print('Image Not Blurred')
        
    #Alul- vagy Túlexponálás felismerése
    exposed = estimate_exposure(image, exposure_thresholds=[0, 51, 102, 153, 204, 256])
    if(exposed == True):
        image = np.array(Histogram_equalization(image))

    #Képek mentése
    cv2.imwrite(image_names[i], image)
    i += 1
    
    
    
