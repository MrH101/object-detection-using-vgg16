#importing modules to be used in developing the video object detection model


import streamlit as st
import tempfile
import cv2    
import math  
import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from PIL import Image
model = VGG16() 

st.title("vgg16 object detection")
def upload_vid():
  vid_upload = st.file_uploader("Upload Video for object classification", type=["mp4","avi"])# SELECTING A SPECIFIC VIDEO TYPE TO UPLOAD
  vids = tempfile.NamedTemporaryFile(delete=False)
  if vid_upload is not None:  
    vids.write(vid_upload.read())
  return vids.name 
      
def framing(videoPath):
  cap = cv2.VideoCapture(videoPath)
  frameRate = cap.get(5) 
  tempImage = tempfile.NamedTemporaryFile(delete=False) 
  x=1
  count = 0
  # Splitting video frames into photos
  while(cap.isOpened()):
    frameId = cap.get(1) 
    ret, frame = cap.read()
    if (ret != True):
      break
    if (frameId % math.floor(frameRate) == 0):
      tempImage = videoPath.split('.')[0] +"_frame%d.jpg" % count;
      count+=1
      cv2.imwrite(tempImage, frame)
      frames.append(tempImage)
      cap.release() 
      return frames,count
    return frames,count


def classifyObjects(): 
  classify = []
  format_string = [] 
  frames,count = framing(videoFile)

  for i in range(count):    
    image = load_img(frames[i], target_size=(224, 224)) 
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)   
    prediction = model.predict(image)  
    label = decode_predictions(prediction)
    label2 = label[0] 
    label3=label2[:4]
    #result =  label2[1]
    label4=[]
    for item in label3:
      label4.append(item[1])
    itemString = listToString(label4)
    return label4
    #format_string.append(result)
  #return format_string

user_input = st.text_input("Enter object to be searched: ")
videoFile = upload_vid()

def compare_frames(object_):
  indexes = []
  label4 = classifyObjects()
  if object_ in label4:
    for i in range(len(label4)):
      if label4[i] == object_:
        index = format_string.index(object_)
        indexes.append(index)
        filePath = frames[index]
        img = load_img(filePath, target_size = (224, 224, 224))
        detected_paths.append(filePath)
    for i in range(len(indexes)):
      st.image(frames[i], width=224)
  else:
    st.write("Object not found!")

if st.button('search'):  
  frames =[]
  detected_paths = []
  compare_frames(user_input)
  st.write("")
