from keras.applications.vgg16 import VGG16
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tensorflow as tf

model = VGG16()
model.summary()


column_names = ["frame", "objects",]
Data = pd.DataFrame(columns = column_names)



def search(query):
  indexList = mainData['objects'].str.contains(query)
  arry = []
  arry = mainData[indexList]['frame']
  print(arry)
  return arry

def addFrameToFile(frame,objects,):
  print("adding to Dataframe")
  new_row = {'frame':frame, 'objects':items,}
  global Data
  Data = Data.append(new_row, ignore_index=True)
  print(Data)

# Function to convert  
def toString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele
        str1 += " , "  
    
    # return string  
    return str1


def createDataFrame():
  print("Dataframe Created")


def videoAnaslysis():
  directory = "Frames"
  for filename in os.listdir(directory):
      if filename.endswith(".jpg"): 
          analyzeFrame(filename)
          continue
      else:
          continue

def convertToFrames(video_loc):
    print("creating Frames...\n")
    cap = cv2.VideoCapture(video_loc)
    success,image = vidcap.read()
    #print(success)
    count = 0
    
    while success:
        cv2.imwrite("images/frame%d.jpg" % count, image)
        # analyzeFrame("Frames/frame%d.jpg" % count)
            # save frame as JPEG file
        sampling_rate=10
        i=0
        while i<sampling_rate:
          success,image = vidcap.read()
          i += 1

        count += 1
        print('Frames: ', count-1)
        
    return True

def predict(frame):
  image = load_img('static/'+frame, target_size=(224, 224))
  image = img_to_array(image)
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  image = preprocess_input(image)
  preprocess_input = model.predict(image)
  label = decode_predictions(pop)
  label3 = label[0]
  label4 = label3[:4]
  label2 = []
  for item in label4:
    label2.append(item[1])
  itemString = listToString(label2)
  # print the classification
  # print('%s (%.2f%%)' % (label2[1], label2[2]*100))
  print(label2[1][1])
  addToDataFrame(frame,itemString)
  return label2