#!/bin/bash

if [ ! -e "./kitti.h5" ]; then
  wget -P "./kitti.h5" "https://s3-eu-west-1.amazonaws.com/densedepth/kitti.h5"
fi

python main.py \
    --host "0.0.0.0" \
    --port 5052
