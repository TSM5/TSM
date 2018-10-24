# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 22:08:07 2018

@author: manik
"""
import cv2
from PIL import Image
import face_recognition as fr
image = face_recognition.load_image_file("..\sun\last.jpeg")
face_locations = face_recognition.face_locations(image)
for face in face_locations:
    top, right, bottom, left = face
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
