from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.engine.topology import Layer
from keras import backend as K
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from utils import LRN2D
import utils
import sys
from OpenFaceModel  import OpenFaceModel

haarcascade_xml_path="haarcascade_frontalface_default.xml"
wait_milisecons=100
image_folder="images/*"
min_distance=200
threshold=0.68

np.set_printoptions(threshold=sys.maxsize)

model=OpenFaceModel()

model.loadWeights()

input_embeddings = model.create_input_image_embeddings(image_folder=image_folder)
model.recognize_faces_in_cam(input_embeddings,
                             haarcascade_xml_path=haarcascade_xml_path,
                             wait_milisecons=wait_milisecons,
                             min_distance=min_distance,
                             threshold=threshold)








