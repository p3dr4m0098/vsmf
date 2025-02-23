import os
import time
import json
import ffmpy
import torch
import signal
import random
import shutil
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from torch.utils.data import Dataset
from mmengine.config import Config, DictAction
from SoccerNet.Downloader import SoccerNetDownloader
from oslactionspotting.core.utils.default_args import (
    get_default_args_dataset,
    get_default_args_model,
)
from oslactionspotting.models.builder import build_model
from oslactionspotting.datasets.builder import build_dataset
from oslactionspotting.apis.inference.builder import build_inferer
from oslactionspotting.apis.inference.utils import search_best_epoch
from oslactionspotting.core.utils.io import check_config, whether_infer_split
from Feature_Extraction.VideoFrameFeatureExtraction import VideoFeatureExtractor, PCAReducer
import tensorflow as tf
import gc

def vision_summarizer(first_video_dir, first_video_name):
    myFeatureExtractor = VideoFeatureExtractor(
        feature="ResNET",
        back_end="TF2",
        transform="crop",
        grabber="opencv",
        FPS=30)
    myPCAReducer = PCAReducer(pca_file="./pca_512_TF2.pkl",
                              scaler_file="./average_512_TF2.pkl")
     
    myFeatureExtractor.extractFeatures(path_video_input=os.path.join(first_video_dir, f"{first_video_name}.mp4"),
                                        path_features_output=os.path.join(first_video_dir, f"{first_video_name}.npy"),
                                        overwrite=True)
    
    myPCAReducer.reduceFeatures(input_features=os.path.join(first_video_dir, f"{first_video_name}.npy"),
                                output_features=os.path.join(first_video_dir, f"{first_video_name}_PCA.npy"),
                                overwrite=True)
    cfg = Config.fromfile('./OSL-ActionSpotting/configs/learnablepooling/json_netvlad++_resnetpca512.py')

    torch.manual_seed(42)
    np.random.seed(42)

    numeric_level = getattr(logging, cfg.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % cfg.log_level)

    os.makedirs(os.path.join(first_video_dir, "logs"), exist_ok=True)
    log_path = os.path.join(
        first_video_dir, "logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    )
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    logging.info("Checking configs files")
    check_config(cfg)

    cfg.infer_split = whether_infer_split(cfg.dataset.test)

    logging.info(cfg)

    start = time.time()
    logging.info("Starting main function")

    weights = None

    if weights is not None:
        if not os.path.exists(weights):
            raise ValueError(f"Specified weights file not found: {weights}")
        cfg.model.load_weights = weights
        logging.info(f"Using specified model weights: {weights}")
    elif cfg.model.load_weights is None:
        if cfg.runner.type == "runner_e2e":
            best_checkpoint_path = os.path.join(cfg.work_dir, "best_checkpoint.pt")
            if not os.path.exists(best_checkpoint_path):
                logging.warning(f"Default checkpoint not found: {best_checkpoint_path}")
            cfg.model.load_weights = best_checkpoint_path
            logging.info(f"Using default model weights: {best_checkpoint_path}")
        else:
            default_path = os.path.join(cfg.work_dir, "model.pth.tar")
            cfg.model.load_weights = default_path
            logging.info(f"Using default model weights: {default_path}")

    model = build_model(
        cfg,
        verbose=False if cfg.runner.type == "runner_e2e" else True,
        default_args=get_default_args_model(cfg),
    )


    classes = [
        "Penalty",
        "Kick-off",
        "Goal",
        "Substitution",
        "Offside",
        "Shots on target",
        "Shots off target",
        "Clearance",
        "Ball out of play",
        "Throw-in",
        "Foul",
        "Indirect free-kick",
        "Direct free-kick",
        "Corner",
        "Yellow card",
        "Red card",
        "Yellow->red card",
    ]

    dataloader_dict = dict(
                num_workers=1,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
            )

    first_vid_dict = dict(
        type="FeatureVideosfromJSON",
        path=os.path.join(first_video_dir, f"{first_video_name}_PCA.npy"),
        data_root=first_video_dir,
        framerate=2,
        window_size=20,
        classes=classes,
        metric="loose",
        results="results_spotting_test",
        dataloader = SimpleNamespace(**dataloader_dict),
    )
    first_vid = SimpleNamespace(**first_vid_dict)

    dataset_infer = build_dataset(
        first_vid, cfg.training.GPU, get_default_args_dataset("test", cfg)
    )

    logging.info("Build inferer")

    inferer = build_inferer(cfg, model)

    logging.info("Start inference on first video")

    _ = inferer.infer(dataset_infer)

    shutil.move(f'OSL-ActionSpotting/outputs/learnablepooling/json_netvlad++_resnetpca512/results_spotting_test/videos/{first_video_name}/{first_video_name}_PCA/results_spotting.json', first_video_dir)

    os.system('rm -rf ./OSL-ActionSpotting/outputs/learnablepooling/json_netvlad++_resnetpca512/results_spotting_test.zip')

    os.system('rm -rf OSL-ActionSpotting/outputs/learnablepooling/json_netvlad++_resnetpca512/results_spotting_test')

    os.system(f'rm -rf ./videos/{first_video_name}/{first_video_name}_PCA ./videos/{first_video_name}/logs')
    
if __name__ == "__main__":
    temp1 = '/home/amdor/VideoSummerization/Models/football/videos/sh_iran_english'
    vision_summarizer(temp1, 'sh_iran_english')

