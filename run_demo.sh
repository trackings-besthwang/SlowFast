#/bin/bash

[ ! -d outputs ] && mkdir -p checkpoints
[ ! -e checkpoints/SLOWFAST_8x8_R50.pkl ] && wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl -O checkpoints/SLOWFAST_8x8_R50.pkl

INPUT=./demo/archery.mp4
OUTPUT=outputs/archery.mp4
mkdir -p outputs

# python tools/run_net.py \
#   --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
#   --opts \
#   DEMO.ENABLE True \
#   DEMO.INPUT_VIDEO "$INPUT" \
#   DEMO.OUTPUT_FILE "$OUTPUT" \
#   DEMO.LABEL_FILE_PATH labels/kinetics_classnames.json \
#   TEST.CHECKPOINT_FILE_PATH checkpoints/SLOWFAST_8x8_R50.pkl \
#   TEST.CHECKPOINT_TYPE caffe2 \
#   TRAIN.ENABLE False

#AVA
python tools/run_net.py \
  --cfg ./demo/AVA/SLOWFAST_32x2_R101_50_50.yaml \
  --opts \
  DEMO.ENABLE True \
  DEMO.INPUT_VIDEO ./demo/archery.mp4 \
  DEMO.OUTPUT_FILE ./outputs/archery.mp4 \
  DEMO.LABEL_FILE_PATH labels/ava_classids.json \
  DEMO.DETECTRON2_CFG  COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml \
  DEMO.DETECTRON2_WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl\
  TEST.CHECKPOINT_FILE_PATH ./weights/SLOWFAST_32x2_R101_50_50.pkl \
  DETECTION.ENABLE True \
  TRAIN.ENABLE False \