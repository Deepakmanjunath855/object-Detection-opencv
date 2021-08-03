#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

nms_threshold = 0.45

cap = cv2.VideoCapture(0)
#cap.set(3,1280)
#cap.set(4,720)
# cap.set(10,150)

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 10.0, (640, 480))

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold = 0.50)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)
    indices = cv2.dnn.NMSBoxes(bbox,confs,0.50,nms_threshold)
    #print(indices)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        distance = (2 * 3.14 * 180) / (w.item()+ h.item() * 360) * 1000 + 3  
        distance=int(distance*2.54)
        #str("{:.2f} Inches".format(distance))
        cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
        #cv2.putText(img,label +" " + confidence,(x,y+20),font,2,(34,139,34),2)
        cv2.putText(img,classNames[classIds[i][0]-1].upper() +" "+str(distance)+"cm",(box[0]+10,box[1]+30),
        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        
    #out.write(img) 
    cv2.imshow('Output',img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
#out.release() 
cv2.destroyAllWindows()


# In[ ]:




