#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imutils

font = cv2.FONT_HERSHEY_SIMPLEX
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
car_classifier = cv2.CascadeClassifier('cars.xml')


import tkinter
from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog

root=Tk()
root.title("Image loader")
root.geometry("550x300+300+150")
root.resizable(width=True,height=True)



    
def open_img():
    filename = filedialog.askopenfilename(title='upload image')
    cam=cv2.imread(filename)
    frames=cv2.resize(cam,(950,700))
    #display loaded image
    img = Image.open(filename)
    img = img.resize((250, 250), Image.ANTIALIAS)  
    img = ImageTk.PhotoImage(img) 
    panel = Label(root, image = img)   
    panel.image = img 
    panel.grid(row = 2)
    
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray,1.3 ,5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray =gray[y:y+h,x:x+w]
        roi_color=frames[y:y+h,x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,204),2)
     
    
        smiles = smile_cascade.detectMultiScale(roi_gray, minNeighbors=20)
        for(sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (55,255,255), 3) 
 
    (regions, _) = hog.detectMultiScale(frames, 
                                            winStride=(4, 4), 
                                            padding=(4, 4), 
                                            scale=1.05) 
    for (x, y, w, h) in regions: 
        cv2.rectangle(frames, (x, y), 
                          (x + w, y + h),  
                          (22, 219, 75),2) 
    cars = car_classifier.detectMultiScale(gray, 1.1, 2)
    for (x,y,w,h) in cars:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (245, 20, 185), 2)     
          
    
    cv2.putText(frames,  'human',  (500,30),   font, 1,  (22, 219, 75),  2,  cv2.LINE_4)
    cv2.putText(frames,  'car',  (400,30),   font, 1,  (245, 20, 185),  2,  cv2.LINE_4)
    cv2.putText(frames,  'face',  (300,30),   font, 1,  (255, 0, 0),  2,  cv2.LINE_4)
    cv2.putText(frames,  'eyes',  (200,30),   font, 1,  (0,0,204),  2,  cv2.LINE_4)
    cv2.putText(frames,  'smile',  (100,30),   font, 1,  (55,255,255),  2,  cv2.LINE_4)
    cv2.imshow('image',frames)

def _quit():
    root.quit()     # stops mainloop
    root.destroy()     
    
    
button = Button(root, text="quit", command=_quit).grid(row=3,columnspan=10)
    
btn = Button(root,text='open image',command=open_img).grid(row=1,columnspan=4)
root.mainloop()


# In[ ]:





# In[ ]:




