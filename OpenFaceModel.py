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
import glob

class OpenFaceModel(object):
    def __init__(self):
        self.model=self.getModel()
        return

    def getModel(self):
        myInput = Input(shape=(96, 96, 3))

        x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)
        x = Lambda(LRN2D, name='lrn_1')(x)
        x = Conv2D(64, (1, 1), name='conv2')(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(192, (3, 3), name='conv3')(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
        x = Activation('relu')(x)
        x = Lambda(LRN2D, name='lrn_2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)

        # Inception3a
        inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
        inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
        inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
        inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
        inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

        inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
        inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
        inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
        inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
        inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

        inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
        inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
        inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
        inception_3a_pool = Activation('relu')(inception_3a_pool)
        inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

        inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
        inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
        inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

        inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

        # Inception3b
        inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
        inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
        inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
        inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
        inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

        inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
        inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
        inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
        inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
        inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

        inception_3b_pool = Lambda(lambda x: x ** 2, name='power2_3b')(inception_3a)
        inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: x * 9, name='mult9_3b')(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)
        inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
        inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
        inception_3b_pool = Activation('relu')(inception_3b_pool)
        inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

        inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
        inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
        inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

        inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

        # Inception3c
        inception_3c_3x3 = utils.conv2d_bn(inception_3b,
                                           layer='inception_3c_3x3',
                                           cv1_out=128,
                                           cv1_filter=(1, 1),
                                           cv2_out=256,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(2, 2),
                                           padding=(1, 1))

        inception_3c_5x5 = utils.conv2d_bn(inception_3b,
                                           layer='inception_3c_5x5',
                                           cv1_out=32,
                                           cv1_filter=(1, 1),
                                           cv2_out=64,
                                           cv2_filter=(5, 5),
                                           cv2_strides=(2, 2),
                                           padding=(2, 2))

        inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
        inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

        inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

        # inception 4a
        inception_4a_3x3 = utils.conv2d_bn(inception_3c,
                                           layer='inception_4a_3x3',
                                           cv1_out=96,
                                           cv1_filter=(1, 1),
                                           cv2_out=192,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(1, 1),
                                           padding=(1, 1))
        inception_4a_5x5 = utils.conv2d_bn(inception_3c,
                                           layer='inception_4a_5x5',
                                           cv1_out=32,
                                           cv1_filter=(1, 1),
                                           cv2_out=64,
                                           cv2_filter=(5, 5),
                                           cv2_strides=(1, 1),
                                           padding=(2, 2))

        inception_4a_pool = Lambda(lambda x: x ** 2, name='power2_4a')(inception_3c)
        inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: x * 9, name='mult9_4a')(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)
        inception_4a_pool = utils.conv2d_bn(inception_4a_pool,
                                            layer='inception_4a_pool',
                                            cv1_out=128,
                                            cv1_filter=(1, 1),
                                            padding=(2, 2))
        inception_4a_1x1 = utils.conv2d_bn(inception_3c,
                                           layer='inception_4a_1x1',
                                           cv1_out=256,
                                           cv1_filter=(1, 1))
        inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

        # inception4e
        inception_4e_3x3 = utils.conv2d_bn(inception_4a,
                                           layer='inception_4e_3x3',
                                           cv1_out=160,
                                           cv1_filter=(1, 1),
                                           cv2_out=256,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(2, 2),
                                           padding=(1, 1))
        inception_4e_5x5 = utils.conv2d_bn(inception_4a,
                                           layer='inception_4e_5x5',
                                           cv1_out=64,
                                           cv1_filter=(1, 1),
                                           cv2_out=128,
                                           cv2_filter=(5, 5),
                                           cv2_strides=(2, 2),
                                           padding=(2, 2))
        inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
        inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

        inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

        # inception5a
        inception_5a_3x3 = utils.conv2d_bn(inception_4e,
                                           layer='inception_5a_3x3',
                                           cv1_out=96,
                                           cv1_filter=(1, 1),
                                           cv2_out=384,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(1, 1),
                                           padding=(1, 1))

        inception_5a_pool = Lambda(lambda x: x ** 2, name='power2_5a')(inception_4e)
        inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: x * 9, name='mult9_5a')(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_5a')(inception_5a_pool)
        inception_5a_pool = utils.conv2d_bn(inception_5a_pool,
                                            layer='inception_5a_pool',
                                            cv1_out=96,
                                            cv1_filter=(1, 1),
                                            padding=(1, 1))
        inception_5a_1x1 = utils.conv2d_bn(inception_4e,
                                           layer='inception_5a_1x1',
                                           cv1_out=256,
                                           cv1_filter=(1, 1))

        inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

        # inception_5b
        inception_5b_3x3 = utils.conv2d_bn(inception_5a,
                                           layer='inception_5b_3x3',
                                           cv1_out=96,
                                           cv1_filter=(1, 1),
                                           cv2_out=384,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(1, 1),
                                           padding=(1, 1))
        inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
        inception_5b_pool = utils.conv2d_bn(inception_5b_pool,
                                            layer='inception_5b_pool',
                                            cv1_out=96,
                                            cv1_filter=(1, 1))
        inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

        inception_5b_1x1 = utils.conv2d_bn(inception_5a,
                                           layer='inception_5b_1x1',
                                           cv1_out=256,
                                           cv1_filter=(1, 1))
        inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

        av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
        reshape_layer = Flatten()(av_pool)
        dense_layer = Dense(128, name='dense_layer')(reshape_layer)
        norm_layer = Lambda(lambda x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)

        # Final Model
        model = Model(inputs=[myInput], outputs=norm_layer)

        return model



################################################################################################################
    def loadWeights(self):
        weights = [
            'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
            'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
            'inception_3a_pool_conv', 'inception_3a_pool_bn',
            'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
            'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
            'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
            'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
            'inception_3b_pool_conv', 'inception_3b_pool_bn',
            'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
            'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
            'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
            'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
            'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
            'inception_4a_pool_conv', 'inception_4a_pool_bn',
            'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
            'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
            'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
            'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
            'inception_5a_pool_conv', 'inception_5a_pool_bn',
            'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
            'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
            'inception_5b_pool_conv', 'inception_5b_pool_bn',
            'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
            'dense_layer'
        ]

        conv_shape = {
            'conv1': [64, 3, 7, 7],
            'conv2': [64, 64, 1, 1],
            'conv3': [192, 64, 3, 3],
            'inception_3a_1x1_conv': [64, 192, 1, 1],
            'inception_3a_pool_conv': [32, 192, 1, 1],
            'inception_3a_5x5_conv1': [16, 192, 1, 1],
            'inception_3a_5x5_conv2': [32, 16, 5, 5],
            'inception_3a_3x3_conv1': [96, 192, 1, 1],
            'inception_3a_3x3_conv2': [128, 96, 3, 3],
            'inception_3b_3x3_conv1': [96, 256, 1, 1],
            'inception_3b_3x3_conv2': [128, 96, 3, 3],
            'inception_3b_5x5_conv1': [32, 256, 1, 1],
            'inception_3b_5x5_conv2': [64, 32, 5, 5],
            'inception_3b_pool_conv': [64, 256, 1, 1],
            'inception_3b_1x1_conv': [64, 256, 1, 1],
            'inception_3c_3x3_conv1': [128, 320, 1, 1],
            'inception_3c_3x3_conv2': [256, 128, 3, 3],
            'inception_3c_5x5_conv1': [32, 320, 1, 1],
            'inception_3c_5x5_conv2': [64, 32, 5, 5],
            'inception_4a_3x3_conv1': [96, 640, 1, 1],
            'inception_4a_3x3_conv2': [192, 96, 3, 3],
            'inception_4a_5x5_conv1': [32, 640, 1, 1, ],
            'inception_4a_5x5_conv2': [64, 32, 5, 5],
            'inception_4a_pool_conv': [128, 640, 1, 1],
            'inception_4a_1x1_conv': [256, 640, 1, 1],
            'inception_4e_3x3_conv1': [160, 640, 1, 1],
            'inception_4e_3x3_conv2': [256, 160, 3, 3],
            'inception_4e_5x5_conv1': [64, 640, 1, 1],
            'inception_4e_5x5_conv2': [128, 64, 5, 5],
            'inception_5a_3x3_conv1': [96, 1024, 1, 1],
            'inception_5a_3x3_conv2': [384, 96, 3, 3],
            'inception_5a_pool_conv': [96, 1024, 1, 1],
            'inception_5a_1x1_conv': [256, 1024, 1, 1],
            'inception_5b_3x3_conv1': [96, 736, 1, 1],
            'inception_5b_3x3_conv2': [384, 96, 3, 3],
            'inception_5b_pool_conv': [96, 736, 1, 1],
            'inception_5b_1x1_conv': [256, 736, 1, 1],
        }


        # Set weights path
        dirPath = './openface_weights'
        fileNames = filter(lambda f: not f.startswith('.'), os.listdir(dirPath))
        paths = {}
        weights_dict = {}

        for n in fileNames:
            paths[n.replace('.csv', '')] = dirPath + '/' + n

        for name in weights:
            if 'conv' in name:
                conv_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
                conv_w = np.reshape(conv_w, conv_shape[name])
                conv_w = np.transpose(conv_w, (2, 3, 1, 0))
                conv_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
                weights_dict[name] = [conv_w, conv_b]
            elif 'bn' in name:
                bn_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
                bn_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
                bn_m = genfromtxt(paths[name + '_m'], delimiter=',', dtype=None)
                bn_v = genfromtxt(paths[name + '_v'], delimiter=',', dtype=None)
                weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]
            elif 'dense' in name:
                dense_w = genfromtxt(dirPath + '/dense_w.csv', delimiter=',', dtype=None)
                dense_w = np.reshape(dense_w, (128, 736))
                dense_w = np.transpose(dense_w, (1, 0))
                dense_b = genfromtxt(dirPath + '/dense_b.csv', delimiter=',', dtype=None)
                weights_dict[name] = [dense_w, dense_b]

        for name in weights:
            if self.model.get_layer(name) != None:
                self.model.get_layer(name).set_weights(weights_dict[name])
            elif self.model.get_layer(name) != None:
                self.model.get_layer(name).set_weights(weights_dict[name])

        return self.model



########################################
    def image_to_embedding(self, image):
        # image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (96, 96))
        img = image[..., ::-1]
        img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
        x_train = np.array([img])
        embedding = self.model.predict_on_batch(x_train)
        return embedding


######################################################
    def recognize_face(self,face_image, input_embeddings,min_distance=200,threshold=0.68):
        embedding = self.image_to_embedding(face_image)

        minimum_distance = min_distance
        name = None

        # Loop over  names and encodings.
        for (input_name, input_embedding) in input_embeddings.items():

            euclidean_distance = np.linalg.norm(embedding - input_embedding)

            print('Euclidean distance from %s is %s' % (input_name, euclidean_distance))

            if euclidean_distance < minimum_distance:
                minimum_distance = euclidean_distance
                name = input_name

        if minimum_distance < threshold:
            return str(name), str(minimum_distance)
        else:
            return None,None


###########################
    def create_input_image_embeddings(self,image_folder="images/*"):
        input_embeddings = {}

        for file in glob.glob(image_folder):
            person_name = os.path.splitext(os.path.basename(file))[0]
            image_file = cv2.imread(file, 1)
            input_embeddings[person_name] = self.image_to_embedding(image_file)

        return input_embeddings


################################
    def recognize_faces_in_cam(self,
                               input_embeddings,
                               haarcascade_xml_path="haarcascade_frontalface_default.xml",
                               wait_milisecons=100,
                               min_distance=200,
                               threshold=0.68):
        cv2.namedWindow("Face Recognizer")
        vc = cv2.VideoCapture(0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        face_cascade = cv2.CascadeClassifier(haarcascade_xml_path)

        while vc.isOpened():
            _, frame = vc.read()
            img = frame
            height, width, channels = frame.shape

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Loop through all the faces detected
            identities = []
            for (x, y, w, h) in faces:
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h

                face_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                identity,distance = self.recognize_face(face_image,
                                               input_embeddings,
                                               min_distance=min_distance,
                                               threshold=threshold)

                if identity is not None:
                    img = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(img, str(identity+" ("+distance+")"), (x1 + 5, y1 - 5), font, 1, (255, 255, 255), 2)

            key = cv2.waitKey(wait_milisecons)
            cv2.imshow("Face Recognizer", img)

            if key == 27:  # exit on ESC
                break
        vc.release()
        cv2.destroyAllWindows()


###########################################
    def sampleFaces(self,
                    haarcascade_xml_path="haarcascade_frontalface_default.xml",
                               wait_milisecons=100,
                            sample_num=10):
        cam = cv2.VideoCapture(0)

        face_detector = cv2.CascadeClassifier(haarcascade_xml_path)

        count = 0
        while (True):
            ret, img = cam.read()
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in faces:
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                count += 1
                # Save the captured image into the datasets folder
                cv2.imwrite("images/User_" + str(count) + ".jpg", img[y1:y2, x1:x2])
                cv2.imshow('image', img)
            k = cv2.waitKey(wait_milisecons) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= sample_num:  # Take 30 face sample and stop video
                break
        cam.release()
        cv2.destroyAllWindows()


