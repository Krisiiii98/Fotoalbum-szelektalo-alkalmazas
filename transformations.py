import tkinter as tk
from tkinter import filedialog
from tkinter import *
import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_arch import PSFPredictor
from torchvision import transforms
from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d
from faults_detection import purple_fringe_detection

MOTION_MODEL_PATH = "models/psf_predictor_motion.pth"
DEFOCUS_MODEL_PATH = "models/psf_predictor_defocus.pth"
KERNEL_SIZE = 15

motion_model = PSFPredictor(kernel_size=15)
motion_model.load_state_dict(torch.load(MOTION_MODEL_PATH, map_location="cpu"))
motion_model.eval()

defocus_model = PSFPredictor(kernel_size=15)
defocus_model.load_state_dict(torch.load(DEFOCUS_MODEL_PATH, map_location="cpu"))
defocus_model.eval()

#Hisztogram kiegyenlítés alul- vagy túlexponált képek javítására
def get_brightness(image):
    """ Meghatározza a kép átlagos fényerejét """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])  # V csatorna átlaga

def apply_gamma_correction(img, gamma):
    """Gamma korrekció alkalmazása a kép világosságának módosítására."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

# Automatikusan alkalmazkodó CLAHE függvény
def exposure_correction(img, apply_gamma=None, gamma_value = 1.0):
    """ Automatikusan alkalmazkodó CLAHE-algoritmus színmegőrzéssel """
    
    brightness = get_brightness(img)

    # Automatikusan döntünk a gamma alkalmazásáról
    if apply_gamma is None:
        if brightness < 80: #Erősen alulexponált, tehát használjuk a gamma korrekciót
            gamma_value = 1.5
            apply_gamma = True
        elif brightness > 180: #Túlexponált, nem használjuk
            apply_gamma = False
        else: #Közepes fényerő, enyhébb gamma korrekció
            gamma_value = 1.2
            apply_gamma = True
    
    # Alapértelmezett CLAHE paraméterek
    clip_limit = 2.0
    tile_grid_size = (8, 8)

    # Paraméterek dinamikus módosítása fényerő alapján
    if brightness < 70:  # Erősen alulexponált
        clip_limit = 4.0
    elif brightness < 120:  # Enyhén alulexponált
        clip_limit = 3.0
    elif brightness > 200:  # Erősen túlexponált
        clip_limit = 1.5
    elif brightness > 160:  # Enyhén túlexponált
        clip_limit = 2.0

    # HSV színtérbe alakítás
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Gamma korrekció V csatornán (ha engedélyezett)
    if apply_gamma:
        v = apply_gamma_correction(v, gamma_value)

    # CLAHE alkalmazása a V csatornára
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v_clahe = clahe.apply(v)

    # Színtelítettség megőrzése: az eredeti és a CLAHE-val módosított kép átlagolása
    v_final = cv2.addWeighted(v, 0.5, v_clahe, 0.5, 0)

    # HSV csatornák újraegyesítése és visszaalakítás BGR-be
    hsv_clahe = cv2.merge((h, s, v_final))
    result = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

    return result

# Képek élesítése

# --- Wiener dekonvolúció ---
def wiener_deconvolution(blurred, kernel, K=0.01):
    h, w = blurred.shape
    kernel = kernel / kernel.sum()
    pad = np.zeros_like(blurred)
    kh, kw = kernel.shape
    pad[:kh, :kw] = kernel

    H = fft2(pad)
    G = fft2(blurred)
    H_conj = np.conj(H)
    F_hat = H_conj * G / (H_conj * H + K)
    result = np.abs(ifft2(F_hat))
    return np.clip(result, 0, 1)

def blur_correction(image, blur_type):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    
    if blur_type == "motion":
        img_tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0)
        with torch.no_grad():
            psf = motion_model(img_tensor)[0, 0].numpy()
            
        img_f = img_rgb.astype(np.float32) / 255.0
        restored = np.zeros_like(img_f)
        for c in range(3):
            restored[:, :, c] = wiener_deconvolution(img_f[:, :, c], psf)
    elif blur_type == "defocus":
        img_tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0)
        with torch.no_grad():
            psf = defocus_model(img_tensor)[0, 0].numpy()
            
        img_f = img_rgb.astype(np.float32) / 255.0
        restored = np.zeros_like(img_f)
        for c in range(3):
            restored[:, :, c] = wiener_deconvolution(img_f[:, :, c], psf)
    restored_uint8 = (restored * 255).astype(np.uint8)
    restored_img = cv2.cvtColor(restored_uint8, cv2.COLOR_RGB2BGR)

    return restored_img
    

#Piros szem javítása
def Red_eye_correction(img, eyeRects):
    for x,y,w,h in eyeRects:
            #a szem területének kivágása
            eyeImage = img[y:y+h , x:x+w]

            #a kép felosztása színcsatornákra(kék, zöld, piros)
            b, g ,r = cv2.split(eyeImage)

            #kék és zöld csatornák összerakása
            bg = cv2.add(b,g)

            #a maszk küszöbértéke a piros kék és zöld színek kombinációja alapján
            mask  = ( (r>(bg-20)) & (r>80) ).astype(np.uint8)*255

            #összes kontúr megkeresése
            contours, _ = cv2. findContours(mask.copy() ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )  # It return contours and Hierarchy

            #a kontúr maximális területének megkeresése
            maxArea = 0
            maxCont = None
            for cont in contours:
                area = cv2.contourArea(cont)
                if area > maxArea:
                    maxArea = area
                    maxCont = cont
            mask = mask * 0  #a mask kép visszaállítása teljesen feketére
            #a legnagyobb kontúr rajzolása a mask-ra
            cv2.drawContours(mask , [maxCont],0 ,(255),-1 )
            #a lyukak lezárása egy sima régió létrehozásához
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_DILATE,(5,5)) )
            mask = cv2.dilate(mask , (3,3) ,iterations=3)

            #a piros színű részek elvesznek,
            #és kitöltjük a helyüket a zöld szín átlagát mindhárom csatornán a textúra fenntartása érdekében
            mean  = (bg / 2).astype(np.uint8)
            mean = mean

            #fekete átlagérték kitöltése a mask képre
            mean = cv2.bitwise_and(mean , mask )  #mean kép maszkolása
            mean  = cv2.cvtColor(mean ,cv2.COLOR_GRAY2BGR ) #mean konvetálása 3 színcsatornára
            mask = cv2.cvtColor(mask ,cv2.COLOR_GRAY2BGR )  #mask konvertálása.....
            eye = cv2.bitwise_and(~mask,eyeImage) + mean           #átlagos szín másolása a maszkolt régióra a színes képen
            img[y:y+h , x:x+w] = eye
            
            return img

#Purple fringe javítása
def purple_fringe_correction(img, mask):
    initial_reduction_step = 0.95
    reduction_decay = 0.98
    max_iteration = 25

    # Először tisztítjuk a maszkot morfológiai műveletekkel
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Először a zárt morfológiát alkalmazzuk, hogy a kis lyukakat eltüntessük
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Majd a nyílt műveletet, hogy eltávolítsuk a kis zajokat

    cv2.imwrite("Mask_after_morph.jpg", mask)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    purple_pixels = np.sum(mask > 200) #fehér pixelek száma

    purple_ratio = (purple_pixels / img.size) * 100
    iteration = 0
    no_change_iteration = 0
    if purple_ratio < 5:
        prev_mask = mask.copy()
        while np.any(mask) and iteration < max_iteration:
            hsv_img[:, :, 0] = np.where(mask > 0, hsv_img[:,:,0] * initial_reduction_step, hsv_img[:,:,0])
            hsv_img[:, :, 1] = np.where(mask > 0, hsv_img[:, :, 1] * initial_reduction_step, hsv_img[:, :, 1])
            hsv_img[:, :, 2] = np.where(mask > 0, hsv_img[:, :, 2] * initial_reduction_step, hsv_img[:, :, 2])

            img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        
            _, mask= purple_fringe_detection(img)

            if np.sum(mask != prev_mask) == 0:
                no_change_iteration += 1
                print(f"Iteráció: {iteration}, nincs változás a maszkban, iteráció vége.")
                if no_change_iteration == 3:
                    break
            prev_mask = mask.copy()
        
            initial_reduction_step = max(0.8, initial_reduction_step * reduction_decay)
            iteration += 1
            print(f"Iteráció: {iteration}, még van lila pixel")
            print(f"Iteráció: {iteration}, hátralévő lila pixelek: {np.sum(mask > 0)}")

            hsv_img = hsv_img.astype(np.uint8)
            result_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    elif purple_ratio > 5 and purple_ratio <= 20:
        blurred = cv2.medianBlur(img, 5)
        img[mask > 0] = blurred[mask > 0]
        result_img = img

    elif purple_ratio > 20:
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        img[mask > 0] = blurred[mask > 0]
        result_img = img
    
    return result_img
