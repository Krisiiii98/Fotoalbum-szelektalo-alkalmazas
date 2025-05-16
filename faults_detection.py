import tkinter as tk
from tkinter import filedialog
from tkinter import *
import os
import cv2
import numpy as np
import math
from PIL import Image
from ultralytics import YOLO
import pandas
from tensorflow.keras.models import load_model

BLUR_MODEL_PATH = "models/blur_classifier_best.keras"
CLASS_NAMES = ['defocus', 'motion']
LAPLACE_THRESHOLD = 1000.0
NOISE_THRESHOLD = 150.0
RESIZE_HEIGHT = 512
BATCH_SIZE = 8

blur_classifier = load_model(BLUR_MODEL_PATH)

def detect_noise_fft(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    spectrum_mean = magnitude_spectrum.mean()
    
    if spectrum_mean > NOISE_THRESHOLD:
        return TRUE
    else:
        return False
   
def classify_blur_types(img):
    resized = np.array([cv2.resize((img * 255).astype(np.uint8), (512, 512)) / 255.0])
    preds = blur_classifier.predict(resized, batch_size=BATCH_SIZE, verbose=0)
    return CLASS_NAMES[np.argmax(preds[0])]

def resize_keep_aspect(image, target_height):
    h, w = image.shape[:2]
    scale = target_height / h
    new_width = int(w * scale)
    return cv2.resize(image, (new_width, target_height))

def laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = resize_keep_aspect(gray, target_height = RESIZE_HEIGHT)
    lap_var = cv2.Laplacian(resized, cv2.CV_64F).var()
    print(lap_var)
    return lap_var

def estimate_blur(image: np.array):
    lap_var = laplacian_variance(image)

    if lap_var < LAPLACE_THRESHOLD:
        detected_type = classify_blur_types(image)
        return True, lap_var, detected_type
    else:
        return False, lap_var, ""

def estimate_exposure(image: np.array):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    max_pixel_brightness_place = int(np.argmax(hist_gray))

    avg_brightness = np.mean(gray)

    dark_pixels = np.sum(gray < 50) / gray.size
    bright_pixels = np.sum(gray > 205) / gray.size
    
    if max_pixel_brightness_place <= 120 and avg_brightness < 100 and dark_pixels > 0.3:
        return True, 1, max_pixel_brightness_place, avg_brightness, dark_pixels
    elif max_pixel_brightness_place > 120 and max_pixel_brightness_place < 160 and 100 <= avg_brightness <= 180:
        return False, 0, max_pixel_brightness_place, avg_brightness, bright_pixels
    elif max_pixel_brightness_place >= 160 and avg_brightness > 180 and bright_pixels > 0.3:
        return True, 2, max_pixel_brightness_place, avg_brightness, bright_pixels
    else:
        return False, 0, max_pixel_brightness_place, avg_brightness, bright_pixels

def estimate_redeye(image: np.array): 
    eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    eyeRects = eyesCascade.detectMultiScale(image , 1.1, 5)
    eyeRects = eyeRects.astype(np.uint8)
    if len(eyeRects) == 0: 
        return False, eyeRects

    detected_redeye = 0

    lowered = np.array([160,100,20])
    uppered = np.array([179,255,255])

    for x,y,w,h in eyeRects:
        eyeImage = image [y:y+h , x:x+w]

        if eyeImage.size == 0:
            continue

        eyeImage_hsv = cv2.cvtColor(eyeImage, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(eyeImage_hsv, lowered, uppered)
        
        croped = cv2.bitwise_and(eyeImage, eyeImage, mask=mask)
  
        redpixel_counter = np.count_nonzero(croped)
        allpixel_count = eyeImage.size
        threshold = allpixel_count * 0.01
        if redpixel_counter > threshold:
            detected_redeye += 1
    
    return detected_redeye > 0, eyeRects

def purple_fringe_detection(image: np.array):
    b, g, r = cv2.split(image)

    diff_rb = np.abs(r - b)

    contrast_threshold = 30
    high_contrast_area = np.abs(r - g) > contrast_threshold

    purple_fringing = np.logical_and(diff_rb > 50, high_contrast_area)
    purple_fringing_image = np.zeros_like(image)
    purple_fringing_image[purple_fringing] = [255, 255, 255]
    purple_fringing_image = cv2.cvtColor(purple_fringing_image, cv2.COLOR_BGR2GRAY)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([160, 255, 255])

    lower_bluepurple = np.array([120, 50, 50])
    upper_bluepurple = np.array([130, 255, 255])

    lower_redpurple = np.array([160, 50, 50])
    upper_redpurple = np.array([180, 255, 255])

    purple_mask = cv2.inRange(hsv_image, lower_purple, upper_purple)
    bluepurple_mask = cv2.inRange(hsv_image, lower_bluepurple, upper_bluepurple)
    redpurple_mask = cv2.inRange(hsv_image, lower_redpurple, upper_redpurple)

    color_mask = cv2.addWeighted(purple_mask, 1, bluepurple_mask, 1, 0)
    color_mask = cv2.addWeighted(color_mask, 1, redpurple_mask, 1, 0)

    combined_mask = cv2.bitwise_and(purple_fringing_image, purple_fringing_image, mask = color_mask)

    if np.any(combined_mask != 0):
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_fringing_mask = np.zeros_like(color_mask)
        area_threshold = image.size * 0.0005

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_threshold:
                continue
            cv2.drawContours(valid_fringing_mask, [contour], -1, 255, thickness=cv2.FILLED)

        contour_mask = cv2.bitwise_and(color_mask, color_mask, mask=valid_fringing_mask)

        purple_pixels = 0
        if np.any(contour_mask != 0):
            purple_pixels = np.count_nonzero(contour_mask)
            return True, contour_mask, purple_pixels
        else:
            return False, contour_mask, purple_pixels
    else:
        purple_pixels = np.count_nonzero(combined_mask)
        return  False, combined_mask, purple_pixels

def face_detection(img):
    face_model = YOLO('models/yolov8n-face.pt')

    face_result = face_model.predict(img, conf=0.40)

    a = face_result[0].boxes.data
    px = pandas.DataFrame(a).astype("float")
    detected_faces = 0
    
    return not px.empty