#!/bin/bash

set -e

LOG="./log/SRDCN_291_31.log"

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

CAFFE=/home/cvpr/caffe/build/tools/caffe	## your caffe path 

$CAFFE train --solver=./solver.prototxt -gpu 0 
