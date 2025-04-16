from PIL import Image as PILImage, ImageStat
import glob
import cv2
import numpy as np
import os, os.path
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *
from tkinter import ttk
from tkinter.ttk import Progressbar
import matplotlib.pyplot as plt
import fnmatch
import imagehash

from faults_detection import purple_fringe_detection, estimate_blur, estimate_exposure, estimate_redeye, face_detection


class ImageStore(object):
    images = []
    image_names = []
    valid_images = [".jpg", ".png", ".jpeg", ".JPG"]
    duplicates = []
    path = ""
    gallery_path = ""
    assistant_path = ""

def folder_creation(path):
    max_position = 0
    position = 0
    image_path = path

    for i in range(len(image_path)):
        if image_path[i:i+1] == "\\":
            position = i    
        if max_position < position:
            max_position = position

    parent_directory = image_path[0:position]

    directory = "Photo Gallery"

    assistant_directory = "Other Images"

    ImageStore.gallery_path = os.path.join(parent_directory, directory)

    ImageStore.assistant_path = os.path.join(parent_directory, assistant_directory)
    
    existing_directory = os.path.isdir(ImageStore.gallery_path)
    existing_directory2 = os.path.isdir(ImageStore.assistant_path)

    if existing_directory is False:
        os.mkdir(ImageStore.gallery_path)

    if existing_directory2 is False:
        os.mkdir(ImageStore.assistant_path)

    subfolders = ['Flowers', 'Landscapes', 'Cars', 'Animals', 'Group of People']
    for i in range(len(subfolders)):
        folder = subfolders[i]
        subfolders_path = os.path.join(ImageStore.gallery_path, folder)
        existing_directory = os.path.isdir(subfolders_path)
        if existing_directory is False:
            os.mkdir(subfolders_path)


def nextPage():
    folderselect.destroy()
    import photoselector

def progressbar_step(file_count):
    step_value = 99.9 / file_count
    progressbar['value'] += step_value
    return progressbar['value']

def upload_images_and_preprocessing():
    currdir = os.getcwd()
    tempdir = filedialog.askdirectory(parent=folderselect, initialdir=currdir, title='Please select a directory')
    ImageStore.path = tempdir
    ImageStore.path = ImageStore.path.replace('/','\\')
    file_count = len(fnmatch.filter(os.listdir(ImageStore.path), '*.*'))
    image_files = [_ for _ in os.listdir(ImageStore.path) if _.endswith('.jpg') or _.endswith('.png') or _.endswith('.JPG') or _.endswith('.jpeg')]
    if len(image_files) == 0:
        messagebox.showerror("Hiba", "Mappában nem található kép! \n Válasszon képeket tartalmazó mappát!")
        upload_images_and_preprocessing()
    folder_creation(ImageStore.path)
    
    checked_images = []
    pg = 0
    for file_org in image_files:
        if file_org not in checked_images:
            dp_images = []; dp_image_quality_score = []; dp_exposed = []; dp_quality = []; dp_blurry = []; dp_edge_count = []; dp_purple_image = []; dp_purple = []; dp_redeye = []; dp_eyerects = []

            #ext = os.path.splitext(file_org)[1]
            #currentfilepath = image.path + "\\" + file_org
            if not file_org in dp_images:
                image_org = PILImage.open(os.path.join(ImageStore.path, file_org))
                hash1 = imagehash.whash(image_org)

                for file_check in image_files:
                    if file_check != file_org:
                        image_check = PILImage.open(os.path.join(ImageStore.path, file_check))
                        hash2 = imagehash.whash(image_check)

                        similiraty = hash1 - hash2
                        if similiraty == 0:
                            dp_images.append(file_org)
                            dp_images.append(file_check)
        
            if len(dp_images) > 0:
                for dp_image_name in dp_images:
                    dp_image = cv2.imread(os.path.join(ImageStore.path, dp_image_name))
                    exposed, quality = estimate_exposure(dp_image)
                    dp_exposed.append(str(exposed))
                    #dp_image_quality_score.append(int(quality))
                    blurry, score, _ = estimate_blur(dp_image)
                    dp_blurry.append(str(blurry))
                    dp_edge_count.append(score)
                    purple_img, purple = purple_fringe_detection(dp_image)
                    dp_purple_image.append(purple_img)
                    dp_purple.append(str(purple))
                    detected_faces = face_detection(dp_image)
                    if detected_faces > 0:
                        redeye, _ = estimate_redeye(dp_image)
                        dp_redeye.append(str(redeye))
                        dp_purple_image.append(purple_img)
                    else:
                        redeye = "False"
                        dp_redeye.append(redeye)

                for i in range(len(dp_images)):
                    point = 0    
                    if dp_exposed[i] == "False":
                        point += 100
                
                    if dp_blurry[i] == "False":
                        point += 100

                    if dp_purple[i] == "False":
                        point += 100 
                
                    if dp_redeye[i] == "False":
                        point += 100  
                
                    dp_image_quality_score.append(point)

                max1 = 0
                max2 = 0
                max1place = 0
                max2place = 0
                index = 0

                for i in dp_image_quality_score:
                    if int(i) > max1:
                        max2 = max1
                        max1 = int(i)
                        max2place = max1place
                        max1place = index
                    elif int(i) > max2:
                        max2 = i
                        max2place = index
                    index += 1
            
                if int(max1) == int(max2):
                    max1 += dp_edge_count[max1place]
                    max2 += dp_edge_count[max2place]
            
                if max1 >= max2:
                    ImageStore.images.append(cv2.imread(os.path.join(ImageStore.path, dp_images[max1place])))
                    ImageStore.image_names.append(dp_images[max1place])
                    for dp_image in dp_images:
                        checked_images.append(dp_image)
                        pg = progressbar_step(file_count)
                        folderselect.update_idletasks()
                elif max2 > max1:
                    ImageStore.images.append(cv2.imread(os.path.join(ImageStore.path, dp_images[max2place])))
                    ImageStore.image_names.append(dp_images[max2place])
                    for dp_image in dp_images:
                        checked_images.append(dp_image)
                        pg = progressbar_step(file_count)
                        folderselect.update_idletasks()
            else:
                ImageStore.images.append(cv2.imread(os.path.join(ImageStore.path, file_org)))
                ImageStore.image_names.append(file_org)
                pg = progressbar_step(file_count)
                folderselect.update_idletasks()

    if round(pg) == 100:
        nextPage()

folderselect = tk.Tk()
folderselect.geometry('300x300')
folderselect.resizable(False, False)
folderselect.title('Fotóalbum szelektáló')
folderselect.configure(background='#CDCDCD')

heading = tk.Label(folderselect, text="Fotóalbum szelektáló", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack(side=TOP)

upload = tk.Button(folderselect, text="Mappa kiválasztása", command=lambda: upload_images_and_preprocessing(), padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'), pady=10)
upload.pack(side=TOP, pady=50)

progressbar = ttk.Progressbar(folderselect, orient='horizontal', length=280)
progressbar.pack(side=TOP, pady=25)

folderselect.mainloop()