#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

#SOLVER=models/pascal_voc/VGG16/faster_rcnn_end2end/solver.prototxt
SOLVER=models/pascal_voc/VGG16/faster_rcnn_end2end/solver_pruned.prototxt
#WEIGHTS=data/imagenet_models/VGG16.v2.caffemodel
#WEIGHTS=data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel
WEIGHTS=data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel.pruned
DATASET="voc_2007_trainval"
ITERS=70000

LOG="experiments/logs/faster_rcnn_end2end_VGG16_.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --solver ${SOLVER} \
  --weights ${WEIGHTS} \
  --imdb ${DATASET} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x
