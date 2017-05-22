#!/bin/bash

MODEL_DIR=/tmp/yt8m
python train.py --train_data_pattern='/path/to/features/train*.tfrecord' --model=LogisticModel --train_dir=$MODEL_DIR/video_level_logistic_model

#train script
#python train.py --train_data_pattern='/media/joyful/HDD/yt8m/video_train/train*.tfrecord' --model=Logistic_Multi_Layer_Model --train_dir=$MODEL_DIR/video_level_multi_layer_logistic_model --feature_names="mean_rgb, mean_audio" --feature_sizes="1024, 128" --label_loss="CenterLoss"