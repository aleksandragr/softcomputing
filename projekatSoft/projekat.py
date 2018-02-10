# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 02:15:35 2018

@author: Saska
"""

import numpy as np
import cv2

import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16,12


from scipy import ndimage
from vector import pnt2line

import os
from keras.models import model_from_json



np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten	
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


from keras.datasets import mnist
 
#dimenzije slike da postavim
from keras import backend as K
K.set_image_dim_ordering('th')

import math


kernel = np.ones((2,2),np.uint8)

elements = []
t =0
counter = 0
times = []
total = 0
i=0;
suma=[]

def loadVideo(video):
    
    cap = cv2.VideoCapture(video)

   
    frames_org = []
    
    while(cap.isOpened):
        
        
        ret, frame = cap.read()
        
        
        frames_org.append(frame)
        if ret==False:
            break
        
        findObject(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
        
        cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
    cap.release()
    cv2.destroyAllWindows()    
    
cc = -1
def nextId():
    
    counter=0
    global cc
    cc += 1
    counter=counter+1
    return cc

def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)

def length(v):
    x,y = v
    return math.sqrt(x*x + y*y)

def distance(p0,p1):
    return length(vector(p0,p1))

def inRange(r, item, items):
    
    retVal = []
    for obj in items:
        ic=item['center']
        oc=obj['center']
        mdist = distance(ic, oc)
        if(mdist<r):
            retVal.append(obj)
    return retVal

stap = 1
def findObject(image):
    
    global total
    global i
    global listaBr
    global stap
    global line
    boundaries = [([230, 230, 230], [255, 255, 255]) ]
    
    img_org = image.copy()
    
    gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    
    (lower, upper) = boundaries[0]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    
  
    mask = cv2.inRange(image, lower, upper)  
    m=1.0*mask
    img0 = m

    img0 = cv2.dilate(img0,kernel) #cv2.erode(img0,kernel)
    img0 = cv2.dilate(img0,kernel)

    labeled, nr_objects = ndimage.label(img0)
    objects = ndimage.find_objects(labeled)
    
    
    if stap == 1:    
        line = houghTrans(gray,image)
        stap=2
    x1=line[0]
    y1=line[1]
    x2=line[2]
    y2=line[3]
    
    
    for i in range(nr_objects):
        
        loc = objects[i]
        (xc,yc) = ((loc[1].stop + loc[1].start)/2,
                   (loc[0].stop + loc[0].start)/2)
        (dxc,dyc) = ((loc[1].stop - loc[1].start),
                   (loc[0].stop - loc[0].start))
    
    
        if(dxc>10 or dyc>10):
        
            x=xc
            y=yc
            dx=dxc
            dy=dyc
            cv2.circle(image, (x,y), 16, (25, 25, 255), 1)            
            elem = {'center':(x,y), 'size':(dx,dy), 't':t}
            

            lst = inRange(20, elem, elements)
            
            nn = len(lst)

            if nn == 0:
                elem['id'] = nextId()
                elem['t'] = t
                elem['pass'] = False
                elem['history'] = [{'center':(xc,yc), 'size':(dxc,dyc), 't':t}]
                elem['number'] = None
                elements.append(elem)
            elif nn == 1:
                lst[0]['center'] = elem['center']
                lst[0]['t'] = t
                lst[0]['history'].append({'center':(xc,yc), 'size':(dxc,dyc), 't':t}) 
                 
                
            
           
    for el in elements:
        et=el['t']
        tt = t - et
        if(tt<3):
            if el['number'] is None:              
                a,b=findReg(img_org,el['center'])
                
                br = returnNum(b)
                
            if videoName == 'video/video-0.avi':
                dist, pnt, r = pnt2line(el['center'], (x1,y1), (x2+10,y2-8))
            else:
                dist, pnt, r = pnt2line(el['center'], (x1,y1), (x2,y2))
           
            i=0;
            if r>0:              
                cv2.line(img_org, pnt, el['center'], (0, 255, 25), 1)
              
                if(dist<9):
                    
                    if el['pass'] == False:
                        el['pass'] = True
                        
                        total += br
                        #print('Broj %d' %br)
                        print('Suma %d' %total)
                        i+=1  
                        
    #print(total)
    
    
def returnNum(pictures):
       
    pictures = np.asarray(pictures)
    
    a = pictures.shape[0]
    pictures = pictures.reshape(a, 1, 28, 28)

    pictures = pictures.astype('float32')
    b = pictures
    pictures = b / 255
    
    image1 = pictures[0:1]
    
    result = model.predict(np.array(image1, np.float32))
    

    vratiB = 0
    for resultt in result:
        maxvalue = np.max(resultt)
        broj = 0
        for res in resultt:
            if res == maxvalue:
                vratiB=broj
                broj = 0
            broj += 1

    return vratiB  
       
def houghTrans(imageG,image_org):

   
    
    edges = cv2.Canny(imageG,50,150,apertureSize = 3)    
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    
    
    l = linesP[1][0]
  
    
          
    
    return l

def houghT(imageG,image_org):

    edges = cv2.Canny(imageG,50,150,apertureSize = 3)
    
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 - 110*(-b))
        y1 = int(y0 - 110*(a))
        x2 = int(x0 - 360*(-b))
        y2 = int(y0 - 360*(a))
        cv2.line(image_org,(x1,y1),(x2,y2),(0,255,0),2)
        
        return x1,y1,x2,y2



def findReg(image_org,centar):
    
    gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    ret,frame_bin = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)   
    picturesRegion = []
      
    x,y=centar
    
    a=y-12
    b=y+12
    c=x-12
    d=x+12
    if videoName == "video/video-5.avi" :
        a=y-13
        b=y+13
        c=x-9
        d=x+9
    else:
        a=y-12
        b=y+12
        c=x-12
        d=x+12
   
    
    region = gray[a:b,c:d]
    
    picturesRegion.append(pictureRegion(region))

           
    return image_org,picturesRegion

def pictureRegion(region): 
    
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def convNet():
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    a_train = X_train.shape[0]
    a_test = X_test.shape[0]
    X_train = X_train.reshape(a_train, 1, 28, 28)
    X_test = X_test.reshape(a_test, 1, 28, 28)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    a = X_train
    b = X_test
    X_train = a / 255
    X_test = b / 255

    j = y_train
    k = y_test
    Y_train = np_utils.to_categorical(j, 10)
    Y_test = np_utils.to_categorical(k, 10)


    model = Sequential()

   
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
    
    
    model.add(Convolution2D(32, 3, 3, activation='relu'))

    
    model.add(MaxPooling2D(pool_size=(2,2)))

    
    model.add(Dropout(0.25))

    
    model.add(Flatten())

    
    model.add(Dense(128, activation='relu'))


    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    model.fit(X_train, Y_train, 
              batch_size=32, nb_epoch=10, verbose=1)



    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)
 


   
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    
    model.save_weights("model.h5")
    print("Saved model to disk")

    return model


def update():
    global x1
    global y1
    if videoName == "video/video-1.avi" :
        x1 = x1 - 10
        y1 = y1 + 10
    
    if videoName == "video/video-6.avi" :
        x1 = x1 - 10
        y1 = y1 + 10
    


if os.path.isfile("model.h5"):
    print("File exists!")
    postoji=True
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model=model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    print("Loaded model from disk")
else:
    print("File does not exist!")
    postoji=False
    model=convNet()


videoName="video/video-5.avi"
#update()
  
  
loadVideo(videoName)
   
 

        
    
    

















