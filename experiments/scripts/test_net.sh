#!/bin/bash

MODEL=models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt
#NET=data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel
#NET=output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_70000.caffemodel
NET=${1}
DATASET="voc_2007_test"
CFG=experiments/cfgs/faster_rcnn_end2end.yml

time ./tools/test_net.py \
  --def ${MODEL} \
  --net ${NET} \
  --imdb ${DATASET} \
  --cfg ${CFG}
