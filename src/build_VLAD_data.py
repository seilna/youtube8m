import os
import sys
import random
import threading

from collections import Counter
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import pudb
import glob
import pudb
from VLAD import *
import utils

tf.flags.DEFINE_string("video_dir", "/dataset/vid_frm",
                       "Video dirctory.")
tf.flags.DEFINE_string("output_dir", "/data1/yj/TFRecord/",
                       "Output TFRecord data directory.")

tf.flags.DEFINE_integer("train_shards", 100,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 10,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 10,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_string("type", "VLAD",
                       "type: RNNFV ,FV, VLAD")

tf.flags.DEFINE_integer("num_threads", 1,
                        "Number of threads to preprocess the videos.")
FLAGS = tf.flags.FLAGS

VideoMetadata = namedtuple("VideoMetadata",
                           ["video_id", "video_path", "information"])

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    """Wrapper for inserting a byte Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting an byte FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _float_feature_list(values):
    """Wrapper for inserting an byte FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _to_sequence_example(cont,vlad):
    """Builds a SequenceExample proto for an video-caption pair etc..

    Args:
        video: An VideoMetadata object.

    Returns:
        A SequenceExample proto.
    """
    pudb.set_trace()
    context = tf.train.Features(feature={
        "video_id": _bytes_feature(cont['video_id']),
        "labels": _int64_feature_list(cont['labels'].values),
        "VLAD": _float_feature(vlad)
    })
    sequence_example = tf.train.Example(features=context)

    return sequence_example


def _process_videos(thread_index, ranges, name, videos, num_shards):
    """Processes and saves a subset of video metadata as TFRecord files in one thread.

    Each thread produces N shards where N = num_shards / num_threads.
    For instance, if num_shards = 128, and num_threads = 2, then the first
    thread would produce shards [0, 64).

    Args:
        thread_index: Integer thread identifier within [0, len(ranges)].
        ranges: A list of pairs of integers specifying the ranges of the datset to
            process in parallel.
        name: Unique identifier specifying the dataset.
        videos: List of VideoMetadata.
        num_shards: Integer number of shards for the output files.
    """

    for i in range(len(videos)):
        vid = videos[i]
        filename_queue = tf.train.string_input_producer([vid], num_epochs=1, shuffle=True)
        reader = tf.TFRecordReader()
        _, serialized_examples = reader.read(filename_queue)
        context_features = {"video_id": tf.FixedLenFeature([], tf.string),
                            "labels": tf.VarLenFeature(tf.int64)}

        sequence_features={"rgb": tf.FixedLenSequenceFeature([], dtype=tf.string),
                        "audio": tf.FixedLenSequenceFeature([], dtype=tf.string)}

        contexts, features = tf.parse_single_sequence_example(
            serialized_examples, context_features=context_features, sequence_features=sequence_features)
        output_file = os.path.join(FLAGS.output_dir, vid.split('/')[-1] )
        writer = tf.python_io.TFRecordWriter(output_file)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            vis_dic = tf.contrib.learn.KMeansClustering(
                num_clusters = 256,
                relative_tolerance=0.0001,
                model_dir='/data1/yj/kmeans/')
            counter = 0
            while not coord.should_stop():
                feat = sess.run(features)
                cont = sess.run(contexts)
                decoded_rgb = tf.reshape(tf.cast(tf.decode_raw(features['rgb'], tf.uint8), tf.float32), shape=[-1,1024])
                temp = sess.run(decoded_rgb)
                vlad = VLAD_tf(utils.Dequantize(temp), vis_dic)
                sequence_example = _to_sequence_example(cont,vlad)
                if sequence_example is not None:
                    counter += 1
                    writer.write(sequence_example.SerializeToString())
            writer.close()
            print("%s [thread %d]: Wrote %d %s working data to %s." %
                (datetime.now(), thread_index, counter, FLAGS.type, output_file))
            sys.stdout.flush()
            shard_counter = 0
        print("%s [thread %d]: Wrote %d %s working data to %d shards." %
            (datetime.now(), thread_index, counter, FLAGS.type, num_shards_per_batch))
        sys.stdout.flush()


def _process_dataset(name, videos, num_shards):
    """Processes a complete data set and saves it as a TFRecord.

    Args:
        name: Unique identifier specifying the dataset.
        videos: List of VideoMetadata.
        num_shards: Integer number of shards for the output files.
    """
    #random.seed(12345)
    #random.shuffle(videos)

    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(videos), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    coord = tf.train.Coordinator()

    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in xrange(len(ranges)):
        args = (thread_index, ranges, name, videos, num_shards)
        t = threading.Thread(target=_process_videos, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)
    print("%s: Finished processing all %d video metadata in the data set '%s'." %
            (datetime.now(), len(videos), name))


def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards)
    assert _is_valid_num_shards(FLAGS.val_shards)
    assert _is_valid_num_shards(FLAGS.test_shards)

    FLAGS.output_dir = os.path.join(FLAGS.output_dir, FLAGS.type)

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    files = glob.glob('/data1/common_datasets/yt8m-data/frame-level/train*.tfrecord')

    _process_dataset("train", files, FLAGS.train_shards)
    #_process_dataset("train_neg", train_neg, FLAGS.train_shards)

    #_process_dataset("val", val_dataset, FLAGS.val_shards)
    #_process_dataset("test", test_dataset, FLAGS.test_shards)


if __name__ == "__main__":
    tf.app.run()
