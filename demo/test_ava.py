from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args

import cv2

import time

import numpy as np
import torch
#import tqdm

from slowfast.utils import logging
from slowfast.visualization.async_predictor import AsyncDemo, AsyncVis, draw_predictions
from slowfast.visualization.demo_loader import VideoManager
from slowfast.visualization.predictor import Predictor
#from slowfast.visualization.video_visualizer import VideoVisualizer, ImgVisualizer

import sys

if __name__ == "__main__":

     args = parse_args()
     path_to_config = args.cfg_files[0]
     cfg = load_config(args, path_to_config)
     cfg = assert_and_infer_cfg(cfg)
     model = Predictor(cfg)
     print("end!!")