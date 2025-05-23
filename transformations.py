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

def get_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])

def apply_gamma_correction(img, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def exposure_correction(img, apply_gamma=None, gamma_value = 1.0):
    
    brightness = get_brightness(img)

    if apply_gamma is None:
        if brightness < 80:
            gamma_value = 1.5
            apply_gamma = True
        elif brightness > 180:
            apply_gamma = False
        else:
            gamma_value = 1.2
            apply_gamma = True

    clip_limit = 2.0
    tile_grid_size = (8, 8)

    if brightness < 70:
        clip_limit = 4.0
    elif brightness < 120:
        clip_limit = 3.0
    elif brightness > 200:
        clip_limit = 1.5
    elif brightness > 160:
        clip_limit = 2.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if apply_gamma:
        v = apply_gamma_correction(v, gamma_value)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v_clahe = clahe.apply(v)

    v_final = cv2.addWeighted(v, 0.5, v_clahe, 0.5, 0)

    hsv_clahe = cv2.merge((h, s, v_final))
    result = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

    return result

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

def get_fine_psf(size=3, motion=True):
    if motion:
        psf = np.zeros((size, size))
        psf[size // 2, size // 2] = 1
        psf = cv2.GaussianBlur(psf, (size, size), 1.0)
    else:
        psf = np.ones((size, size)) / (size * size)
    return psf

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
            
        psf = get_fine_psf(size=5, motion=True)

        img_f = img_rgb.astype(np.float32) / 255.0
        restored = np.zeros_like(img_f)
        for c in range(3):
            restored[:, :, c] = wiener_deconvolution(img_f[:, :, c], psf)
    
    elif blur_type == "defocus":
        img_tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0)
        with torch.no_grad():
            psf = defocus_model(img_tensor)[0, 0].numpy()

        psf = get_fine_psf(size=5, motion=False)

        img_f = img_rgb.astype(np.float32) / 255.0
        restored = np.zeros_like(img_f)
        for c in range(3):
            restored[:, :, c] = wiener_deconvolution(img_f[:, :, c], psf)

    restored_uint8 = (restored * 255).astype(np.uint8)
    restored_img = cv2.cvtColor(restored_uint8, cv2.COLOR_RGB2BGR)

    return restored_img

def Red_eye_correction(img, eyeRects):
    for x,y,w,h in eyeRects:
            eyeImage = img[y:y+h , x:x+w]

            b, g ,r = cv2.split(eyeImage)

            bg = cv2.add(b,g)

            mask  = ( (r>(bg-20)) & (r>80) ).astype(np.uint8)*255

            contours, _ = cv2. findContours(mask.copy() ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )

            maxArea = 0
            maxCont = None
            for cont in contours:
                area = cv2.contourArea(cont)
                if area > maxArea:
                    maxArea = area
                    maxCont = cont
            mask = mask * 0
            cv2.drawContours(mask , [maxCont],0 ,(255),-1 )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_DILATE,(5,5)) )
            mask = cv2.dilate(mask , (3,3) ,iterations=3)

            mean  = (bg / 2).astype(np.uint8)
            mean = mean

            mean = cv2.bitwise_and(mean , mask )
            mean  = cv2.cvtColor(mean ,cv2.COLOR_GRAY2BGR )
            mask = cv2.cvtColor(mask ,cv2.COLOR_GRAY2BGR )
            eye = cv2.bitwise_and(~mask,eyeImage) + mean
            img[y:y+h , x:x+w] = eye
            
            return img

def purple_fringe_correction(img, mask):
    initial_reduction_step = 0.95
    reduction_decay = 0.98
    max_iteration = 25

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    purple_pixels = np.sum(mask > 200)

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
        
            _, mask, _= purple_fringe_detection(img)

            if np.sum(mask != prev_mask) == 0:
                no_change_iteration += 1
                if no_change_iteration == 3:
                    break
            prev_mask = mask.copy()
        
            initial_reduction_step = max(0.8, initial_reduction_step * reduction_decay)
            iteration += 1

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
