from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args

import time
import sys
import numpy as np
import torch

import slowfast.models.optimizer as optim
from slowfast.datasets import loader
from slowfast.models import build_model

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du

from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter

from train_net import eval_epoch

def main():
    sys.argv = [
        "tools/run_net.py",  # argv[0]는 프로그램 이름 (아무거나 넣어도 됨)
        "--cfg", "demo/AVA/SLOWFAST_32x2_R101_50_50.yaml",
        "--opts",
        "DATA.PATH_TO_DATA_DIR", "D:\\works\\AVA",
        "TRAIN.ENABLE", "True",
        "NUM_GPUS", "1",
        "TRAIN.BATCH_SIZE", "8",
        "TRAIN.CHECKPOINT_TYPE", 'pytorch',
        "TRAIN.CHECKPOINT_FILE_PATH", "./weights/SLOWFAST_32x2_R101_50_50.pkl",
        "AVA.TRAIN_LISTS", "['train_short.csv']",
        "AVA.TEST_LISTS", "['val_short.csv']",
        "AVA.GROUNDTRUTH_FILE", 'ava_val_v2.2_short.csv',
        "AVA.TRAIN_GT_BOX_LISTS", "['ava_train_v2.2_short.csv']",
        "AVA.TEST_PREDICT_BOX_LISTS", "['ava_val_v2.2_short.csv']"
    ]
    args = parse_args()

    path_to_config = args.cfg_files[0]
    cfg = load_config(args, path_to_config)
    cfg = assert_and_infer_cfg(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    checkpoint_epoch = cu.load_checkpoint(
        cfg.TRAIN.CHECKPOINT_FILE_PATH,
        model,
        cfg.NUM_GPUS > 1,
        optimizer,
        scaler if cfg.TRAIN.MIXED_PRECISION else None,
        inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
        convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
        clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
        image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
    )
    start_epoch = checkpoint_epoch + 1
    cur_epoch = start_epoch

    val_loader = loader.construct_loader(cfg, "val")
    train_loader = loader.construct_loader(cfg, "train")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )
    val_meter = AVAMeter(len(val_loader), cfg, mode="val")

    eval_epoch(val_loader,model, val_meter, cur_epoch, cfg, train_loader,None)

if __name__ == "__main__":
    main()