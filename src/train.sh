#!/bin/bash

MODEL_DIR=/tmp/yt8m
python train.py --train_data_pattern='/path/to/features/train*.tfrecord' --model=LogisticModel --train_dir=$MODEL_DIR/video_level_logistic_model
