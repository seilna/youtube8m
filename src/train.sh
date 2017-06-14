#!/bin/bash

MODEL_DIR=/tmp/yt8m
python train.py --train_data_pattern='/path/to/features/train*.tfrecord' --frame_features=True --feature_names="rgb, audio" --feature_sizes=1024, 128" --model=ContextMemoryModel --train_dir=$MODEL_DIR/BasicContextMemory --batch_size=128 --base_learning_rate=0.0006 --lstm_cells=1152 --regularizer_penalty=20.0 --learning_rate_decay_examples=2000000 --label_loss=Huber_CrossEntropyLoss --delta=0.5

