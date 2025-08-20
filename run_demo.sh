#/bin/bash

[ ! -d outputs ] && mkdir -p checkpoints
[ ! -e checkpoints/SLOWFAST_8x8_R50.pkl ] && wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl -O checkpoints/SLOWFAST_8x8_R50.pkl

INPUT=./demo/archery.mp4
OUTPUT=outputs/archery.mp4
mkdir -p outputs

python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  --opts \
  DEMO.ENABLE True \
  DEMO.INPUT_VIDEO "$INPUT" \
  DEMO.OUTPUT_FILE "$OUTPUT" \
  DEMO.LABEL_FILE_PATH labels/kinetics_classnames.json \
  TEST.CHECKPOINT_FILE_PATH checkpoints/SLOWFAST_8x8_R50.pkl \
  TEST.CHECKPOINT_TYPE caffe2 \
  TRAIN.ENABLE False