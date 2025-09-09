from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args

import time

import numpy as np
import torch
from slowfast.datasets import loader
from slowfast.datasets.build import build_dataset
from slowfast.datasets import utils as utils


"""
setting pycharm for run 
--cfg
./demo/AVA/SLOWFAST_32x2_R101_50_50.yaml
--opts
DATA.PATH_TO_DATA_DIR
"D:\\works\\AVA"
TRAIN.ENABLE
True
NUM_GPUS
1
TRAIN.BATCH_SIZE
32
AVA.TRAIN_LISTS
['train_short.csv']
AVA.TRAIN_GT_BOX_LISTS
['ava_train_v2.2_short.csv']
"""

def main():
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

        split = 'train'
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

        # Construct the dataset
        dataset = build_dataset(dataset_name, cfg, split)
        #d = dataset[0]
        print(f"data: {dataset[0]}")

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=loader.detection_collate if cfg.DETECTION.ENABLE else None,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )

        for cur_iter, (inputs, labels, index, time, meta) in enumerate(train_loader):
            print(f"cur_iter:{cur_iter}")
            print(f"inputs.shape={inputs[0].shape},{inputs[1].shape}")
            print(f"labels.shape={labels.shape}")
            print(f"index.shape={index.shape}")
            print(f"time.shape={time.shape}")
            print(f"meta[boxes].shape={meta['boxes'].shape}, meta[ori_boxes].shape={meta['ori_boxes'].shape}, meta[metadata].shape={meta['metadata'].shape}")

if __name__ == "__main__":
    print('hello')
    main()
