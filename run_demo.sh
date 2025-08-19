#/bin/bash
# python tools/run_net.py \
#   --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
#   --opts \
#     DEMO.ENABLE True \
#     DEMO.THREAD_ENABLE True \
#     DEMO.LABEL_FILE_PATH labels/kinetics_classnames.json \
#     DEMO.INPUT_VIDEO ./videos/k600/washing_hair/val/2CoZMzjn1HE_000119_000129.mp4 \
#     DEMO.OUTPUT_FILE out.mp4 \
#     TRAIN.ENABLE False \
#     TEST.ENABLE False \
#     TEST.CHECKPOINT_FILE_PATH ./weights/SLOWFAST_8x8_R50.pyth \
#     NUM_GPUS 1

[ ! -d outputs ] && mkdir -p checkpoints
[ ! -e checkpoints/SLOWFAST_8x8_R50.pkl ] && wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl      -O checkpoints/SLOWFAST_8x8_R50.pkl

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