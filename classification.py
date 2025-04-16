from ultralytics import YOLO
import numpy as np
import pandas
import cv2

#Osztályok
classes = { 0:'astilbe',
            1:'audi',
            2:'beach',      
            3:'bellflower',       
            4:'black_eyes_susan',      
            5:'bmw',    
            6:'buildings',      
            7:'calendula',     
            8:'california_poppy',    
            9:'carnation',     
           10:'cat',   
           11:'common_daisy',     
           12:'coreopsis',     
           13:'daffodil',    
           14:'dandelion',     
           15:'dog',       
           16:'group',       
           17:'iris',       
           18:'magnolia',       
           19:'mercedes',     
           20:'mountain',      
           21:'rose',   
           22:'seashore',      
           23:'street',     
           24:'sunflower',       
           25:'tulip',  
           26:'waterlilly', }

class mainclasses:
    Flowers = [0, 3, 4, 7, 8, 9, 11, 12, 13, 14, 17, 18, 21, 24, 25, 26]
    Cars = [1, 5, 19]
    Landscapes = [2, 6, 20, 22, 23]
    Animals = [10, 15]
    Group = 16

def classify(path, image):
    model = YOLO('models/best.pt')
    results = model(image)
    names_dict = results[0].names
    probs = results[0].probs.top1
    print(probs)
    for i in range(len(mainclasses.Flowers)):
        if probs == mainclasses.Flowers[i]:
            path = path + '\\Flowers'
            return path
    for i in range(len(mainclasses.Cars)):
        if probs == mainclasses.Cars[i]:
            path = path + '\\Cars'
            return path
    for i in range(len(mainclasses.Landscapes)):
        if probs == mainclasses.Landscapes[i]:
            path = path + '\\Landscapes'
            return path
    for i in range(len(mainclasses.Animals)):
        if probs == mainclasses.Animals[i]:
            path = path + '\\Animals'
            return path
    if probs == 16:
        path = path + '\\Group of People'
        return path 


