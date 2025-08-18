#/bin/bash
python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  --opts \
    DEMO.ENABLE True \
    DEMO.LABEL_FILE_PATH labels/kinetics_classnames.json \
    DEMO.INPUT_VIDEO ./demo/archery.mp4 \
    DEMO.OUTPUT_FILE out.mp4 \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    TEST.CHECKPOINT_FILE_PATH ./weights/SLOWFAST_8x8_R50.pyth \
    NUM_GPUS 1