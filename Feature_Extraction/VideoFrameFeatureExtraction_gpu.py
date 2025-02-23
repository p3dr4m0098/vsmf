from tqdm import tqdm
import numpy as np
import cv2
import os
import logging
import gc
import argparse
import SoccerNet

import configparser
import math
try:
    # pip install tensorflow (==2.3.0)
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.resnet import preprocess_input
    # from tensorflow.keras.preprocessing.image import img_to_array
    # from tensorflow.keras.preprocessing.image import load_img
    from tensorflow import keras
    import tensorflow as tf
    from tensorflow.data import Dataset
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

except:
    print("issue loading TF2")
    pass

# import argparse
import imutils  # pip install imutils
import skvideo.io
import pickle as pkl

from sklearn.decomposition import PCA, IncrementalPCA  # pip install scikit-learn
from sklearn.preprocessing import StandardScaler
import json

import random
from SoccerNet.utils import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.DataLoader import Frame, FrameCV


def resize_frame_with_padding(frame, target_size=(224, 224)):
    original_height, original_width = frame.shape[:2]
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(target_size[1] * aspect_ratio)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    padded_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    top_padding = (target_size[1] - new_height) // 2
    bottom_padding = target_size[1] - new_height - top_padding
    left_padding = (target_size[0] - new_width) // 2
    right_padding = target_size[0] - new_width - left_padding
    padded_frame[top_padding:top_padding+new_height, left_padding:left_padding+new_width] = resized_frame
    
    return padded_frame



class VideoFrameLoader:
    def __init__(self, video_path, FPS=2.0, transform=None, start=None, duration=None):
        self.video_path = video_path
        self.FPS = FPS
        self.transform = transform
        self.start = start
        self.duration = duration

        self.video_capture = cv2.VideoCapture(self.video_path)
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


        self.frames = []
        self.time_second = self.frame_count / self.fps

    def read_frames(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame = resize_frame_with_padding(frame)
            yield frame

    def close(self):
        self.video_capture.release()


class VideoFeatureExtractor:
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
            base_model = keras.applications.resnet.ResNet152(include_top=True,
                                                             weights='./resnet152_weights_tf_dim_ordering_tf_kernels.h5',
                                                             input_tensor=None,
                                                             input_shape=None,
                                                             pooling=None,
                                                             classes=1000)

            self.model = Model(base_model.input,
                               outputs=[base_model.get_layer("avg_pool").output])
            self.model.trainable = False

    def process_batch(self, batch_frames):
        frames = preprocess_input(np.array(batch_frames))
        batch_features = self.model.predict(frames, batch_size=len(batch_frames), verbose=0)
        return batch_features

    def extractFeatures(self, path_video_input, path_features_output, start=None, duration=None, overwrite=False):
        logging.info(f"Extracting features for video {path_video_input}")

        if os.path.exists(path_features_output) and not overwrite:
            logging.info("Features already exist, use overwrite=True to overwrite them. Exiting.")
            return

        if "TF2" in self.back_end:
            video_loader = VideoFrameLoader(path_video_input, FPS=self.FPS, transform=self.transform, start=start, duration=duration)

            all_features = []

            batch_frames = []
            batch_size = 64
            for frame in tqdm(video_loader.read_frames(), desc="Processing frames"):
                batch_frames.append(frame)

                if len(batch_frames) == batch_size:
                    batch_features = self.process_batch(batch_frames)
                    all_features.append(batch_features)
                    batch_frames = []

            if len(batch_frames) > 0:
                batch_features = self.process_batch(batch_frames)
                all_features.append(batch_features)

            all_features = np.vstack(all_features)

            logging.info(f"Extracted features {all_features.shape}")

        os.makedirs(os.path.dirname(path_features_output), exist_ok=True)
        np.save(path_features_output, all_features)
        logging.info(f"Features saved to {path_features_output}")

        video_loader.close()
        tf.keras.backend.clear_session()
        gc.collect()



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
