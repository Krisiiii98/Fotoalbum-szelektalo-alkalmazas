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


class ImageStore:
    path = ""
    images = []
    image_names = []

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

def detect_noise_fft(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    mean_mag = np.mean(magnitude)
    return mean_mag

def denoise_image(image, method="combined"):
    if method not in ["gaussian", "median", "bilateral", "nlm", "combined"]:
        raise ValueError(f"Ismeretlen módszer: {method}")

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image.copy()
    denoised = np.zeros_like(img_rgb)

    if method in ["gaussian", "median", "bilateral"]:
        for c in range(3):
            channel = img_rgb[:, :, c]
            if method == "gaussian":
                denoised[:, :, c] = cv2.GaussianBlur(channel, (5, 5), 0)
            elif method == "median":
                denoised[:, :, c] = cv2.medianBlur(channel, 5)
            elif method == "bilateral":
                denoised[:, :, c] = cv2.bilateralFilter(channel, 9, 75, 75)
    elif method == "nlm":
        for c in range(3):
            denoised[:, :, c] = cv2.fastNlMeansDenoising(img_rgb[:, :, c], h=10)
    elif method == "combined":
        temp = np.zeros_like(img_rgb)
        for c in range(3):
            temp[:, :, c] = cv2.bilateralFilter(img_rgb[:, :, c], 9, 75, 75)
        for c in range(3):
            denoised[:, :, c] = cv2.fastNlMeansDenoising(temp[:, :, c], h=8)

    return cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)

def select_and_denoise(image):
    score = detect_noise_fft(image)

    if score > 120:
        method = "median"
    elif score > 80:
        method = "bilateral"
    elif score > 60:
        method = "gaussian"
    else:
        method = "nlm"

    return denoise_image(image, method=method)

class ImageScorer:
    def __init__(self, image):
        self.image = select_and_denoise(image)
        self.exposed, self.quality, self.brightness_place, self.avg_brightness, self.dark_or_bright_pixels = estimate_exposure(self.image)
        self.blurry, self.edge_score, _ = estimate_blur(self.image)
        self.purple, _, self.purple_pixels = purple_fringe_detection(self.image)
        self.faces = face_detection(self.image)
        self.redeye = estimate_redeye(self.image)[0] if self.faces > 0 else "False"

    def to_dict(self):
        return {
            "exposed": self.exposed,
            "quality": self.quality,
            "blurry": self.blurry,
            "purple": self.purple,
            "redeye": self.redeye,
            "edge_score": self.edge_score,
            "purple_pixels": self.purple_pixels,
            "brightness_place": self.brightness_place,
            "dark_or_bright_pixels": self.dark_or_bright_pixels,
        }

def process_image_list(image_names, base_path):
    scorers = []
    for name in image_names:
        img_path = os.path.join(base_path, name)
        img = cv2.imread(img_path)
        scorer = ImageScorer(img)
        scorers.append((name, scorer))
    return scorers

def choose_best_image(scorers):
    if len(scorers) == 1:
        return scorers[0][0]

    features = [s.to_dict() for _, s in scorers]
    scores = [0] * len(scorers)

    def assign_score(attribute, good_value, key_func=None, inverse=False):
        values = [f[attribute] for f in features]
        good_indices = [i for i, v in enumerate(values) if v == good_value]

        if good_indices and len(good_indices) < len(values):
            best = max(good_indices, key=lambda i: key_func(features[i]) if key_func else 0)
            scores[best] += 100
        elif not good_indices:
            best = max(range(len(features)), key=lambda i: key_func(features[i]) if key_func else 0)
            scores[best] += 100

    def exposed_key(f):
        return -f["dark_or_bright_pixels"]

    def purple_key(f):
        return -f["purple_pixels"]

    assign_score("exposed", "False", exposed_key)
    assign_score("blurry", "False", lambda f: f["edge_score"])
    assign_score("purple", "False", purple_key)
    assign_score("redeye", "False")

    max_score = max(scores)
    top_candidates = [i for i, s in enumerate(scores) if s == max_score]

    def tie_breaker(i):
        f = features[i]
        return (
            -f["dark_or_bright_pixels"],
            f["edge_score"],
            -f["purple_pixels"]
        )

    best = sorted(top_candidates, key=tie_breaker)[0]
    return scorers[best][0]


def upload_images_and_preprocessing():
    currdir = os.getcwd()
    tempdir = filedialog.askdirectory(parent=folderselect, initialdir=currdir, title='Please select a directory')
    ImageStore.path = tempdir.replace('/', '\\')
    file_count = len(fnmatch.filter(os.listdir(ImageStore.path), '*.*'))
    image_files = [f for f in os.listdir(ImageStore.path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        messagebox.showerror("Hiba", "Mappában nem található kép! \n Válasszon képeket tartalmazó mappát!")
        upload_images_and_preprocessing()
        return

    folder_creation(ImageStore.path)
    checked_images = []

    for file_org in image_files:
        if file_org not in checked_images:
            dp_images = []
            image_org = PILImage.open(os.path.join(ImageStore.path, file_org))
            hash1 = imagehash.whash(image_org)

            for file_check in image_files:
                if file_check != file_org:
                    image_check = PILImage.open(os.path.join(ImageStore.path, file_check))
                    hash2 = imagehash.whash(image_check)
                    if hash1 - hash2 < 5:
                        dp_images.extend([file_org, file_check])

            if dp_images:
                scorers = process_image_list(dp_images, ImageStore.path)
                best_img_name = choose_best_image(scorers)
                ImageStore.images.append(cv2.imread(os.path.join(ImageStore.path, best_img_name)))
                ImageStore.image_names.append(best_img_name)
                for dp_image in dp_images:
                    checked_images.append(dp_image)
                    pg = progressbar_step(file_count)
                    folderselect.update_idletasks()
            else:
                img = cv2.imread(os.path.join(ImageStore.path, file_org))
                img = select_and_denoise(img)
                ImageStore.images.append(img)
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