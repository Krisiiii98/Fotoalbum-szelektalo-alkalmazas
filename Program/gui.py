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
    image = np.array(image_resize(image))

#Képek vizsgálata (képalkotási hibák detektálása)
gui.destroy()
for image in images:
    #Elmosódások felismerése
    blurry = estimate_blur(image, threshold=args.threshold)
    if(blurry == "true"):
        print('Image is Blurred')
    else:
        print('Image Not Blurred')
        
    #Alul- vagy Túlexponálás felismerése
    exposed = estimate_exposure(image, exposure_thresholds=args.threshold)
    
    
    
