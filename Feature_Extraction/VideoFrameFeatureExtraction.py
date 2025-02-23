import argparse
import os
import SoccerNet

import logging

import configparser
import math
try:
    # pip install tensorflow (==2.3.0)
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.resnet import preprocess_input
    # from tensorflow.keras.preprocessing.image import img_to_array
    # from tensorflow.keras.preprocessing.image import load_img
    from tensorflow import keras
except:
    print("issue loading TF2")
    pass
import os
# import argparse
import numpy as np
import cv2  # pip install opencv-python (==3.4.11.41)
import imutils  # pip install imutils
import skvideo.io
from tqdm import tqdm
import pickle as pkl

from sklearn.decomposition import PCA, IncrementalPCA  # pip install scikit-learn
from sklearn.preprocessing import StandardScaler
import json

import random
from SoccerNet.utils import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.DataLoader import Frame, FrameCV




class VideoFeatureExtractor():
    def __init__(self,
                 feature="ResNET",
                 back_end="TF2",
                 overwrite=False,
                 transform="crop",
                 grabber="opencv",
                 FPS=2.0,
                 split="all"):

        self.feature = feature
        self.back_end = back_end
        self.verbose = True
        self.transform = transform
        self.overwrite = overwrite
        self.grabber = grabber
        self.FPS = FPS
        self.split = split


        if "TF2" in self.back_end:

            # create pretrained encoder (here ResNet152, pre-trained on ImageNet)
            base_model = keras.applications.resnet.ResNet152(include_top=True,
                                                             weights='resnet152_weights_tf_dim_ordering_tf_kernels.h5',
                                                             input_tensor=None,
                                                             input_shape=None,
                                                             pooling=None,
                                                             classes=1000)

            # define model with output after polling layer (dim=2048)
            self.model = Model(base_model.input,
                               outputs=[base_model.get_layer("avg_pool").output])
            self.model.trainable = False

        
    def extractFeatures(self, path_video_input, path_features_output, start=None, duration=None, overwrite=False):
        logging.info(f"extracting features for video {path_video_input}")

        if os.path.exists(path_features_output) and not overwrite:
            logging.info("Features already exists, use overwrite=True to overwrite them. Exiting.")
            return
        if "TF2" in self.back_end:

            if self.grabber == "skvideo":
                videoLoader = Frame(
                    path_video_input, FPS=self.FPS, transform=self.transform, start=start, duration=duration)
            elif self.grabber == "opencv":
                videoLoader = FrameCV(
                    path_video_input, FPS=self.FPS, transform=self.transform, start=start, duration=duration)


            if duration is None:
                duration = videoLoader.time_second


            batch_size = 64  # Adjust according to your GPU memory
            features = []
            for i in range(0, len(videoLoader.frames), batch_size):
                batch_frames = preprocess_input(videoLoader.frames[i:i + batch_size])
                batch_features = self.model.predict(batch_frames, batch_size=batch_size, verbose=1)
                features.append(batch_features)

            features = np.concatenate(features, axis=0)


        # save the featrue in .npy format
        np.save(path_features_output, features)



class PCAReducer():
    def __init__(self, pca_file=None, scaler_file=None):
        self.pca_file = pca_file
        self.scaler_file = scaler_file
        self.loadPCA()
    
    def loadPCA(self):
        # Read pre-computed PCA
        self.pca = None
        if self.pca_file is not None:
            with open(self.pca_file, "rb") as fobj:
                self.pca = pkl.load(fobj)

        # Read pre-computed average
        self.average = None
        if self.scaler_file is not None:
            with open(self.scaler_file, "rb") as fobj:
                self.average = pkl.load(fobj)

    def reduceFeatures(self, input_features, output_features, overwrite=False):
        logging.info(f"reducing features {input_features}")

        if os.path.exists(output_features) and not overwrite:
            logging.info(
                "Features already exists, use overwrite=True to overwrite them. Exiting.")
            return
        feat = np.load(input_features)
        if self.average is not None:
            feat = feat - self.average
        if self.pca is not None:
            feat = self.pca.transform(feat)
        np.save(output_features, feat)



