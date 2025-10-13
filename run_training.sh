#/bin/bash

#AVA
python tools/run_net.py \
  --cfg ./demo/AVA/SLOWFAST_32x2_R101_50_50.yaml \
  --opts \
  TRAIN.ENABLE True \
  DATA.PATH_TO_DATA_DIR  "/home/developer/AVA" \
  NUM_GPUS 1 \
  DEMO.ENABLE False \
  TRAIN.BATCH_SIZE 4 
