import streamlit as st
import tempfile
import cv2    
import math  
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from PIL import Image

st.title("Object Detection Using VGG16")

def uploadFile():
  vid_file = st.file_uploader("Upload a video", type=["mp4", "mov","avi"])
  tempVideo = tempfile.NamedTemporaryFile(delete=False) 
  if vid_file is not None: 
    tempVideo.write(vid_file.read())
  return tempVideo.name

def splitVideo(videoPath):
  count = 0
  cap = cv2.VideoCapture(videoPath)
  frameRate = cap.get(5) 
  tempImage = tempfile.NamedTemporaryFile(delete=False) 
  x=1

  while (cap.isOpened()):
    frameId = cap.get(1) 
    ret, frame = cap.read()
    if (ret != True):
      break
    if (frameId % math.floor(frameRate) == 0):
      tempImage = videoPath.split('.')[0] +"_frame%d.jpg" % count;count+=1
      cv2.imwrite(tempImage, frame)
      frames.append(tempImage)
  cap.release() 
  return frames,count

def classifyObjects():  
  model = VGG16()
  classify = []
  frames,count = splitVideo(videoFile)
  print(count)

  for i in range(count):    
    image = load_img(frames[i], target_size=(224, 224)) 
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)   
    img_pred = model.predict(image)
    label = decode_predictions(img_pred)    
    label = label[0][0]
    result =  label[1]
    classify.append(result)
  return classify

def searchInFrames(object_):
  indeces = []
  classifications = classifyObjects()
  if object_ in classifications:
    for i in range(len(classifications)):
      if classifications[i] == object_:
        index = classifications.index(object_)
        indeces.append(index)
        filePath = frames[index]
        img = load_img(filePath, target_size = (224, 224))
        detected_paths.append(filePath)
    for i in range(len(indeces)):
      st.image(frames[i], width=224)
  else:
    st.write("The object NOT FOUND.")

videoFile = uploadFile()
user_input = st.text_input("Enter the object you are searching: ")

if st.button('Search'):
  classifyObjects(videoFile)  
  frames =[]
  detected_paths = []
  searchInFrames(user_input)
  st.write("")
