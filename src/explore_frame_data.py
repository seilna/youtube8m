import tensorflow as tf
from glob import glob
import pudb
import numpy as np
import utils
files = glob('/data1/common_datasets/yt8m-data/frame-level/train*.tfrecord')[::10]
print len(files)
filename_queue = tf.train.string_input_producer(files, num_epochs=1, shuffle=True)

reader = tf.TFRecordReader()
_, serialized_examples = reader.read(filename_queue)

context_features = {"video_id": tf.FixedLenFeature([], tf.string),
                    "labels": tf.VarLenFeature(tf.int64)}

sequence_features={"rgb": tf.FixedLenSequenceFeature([], dtype=tf.string),
                  "audio": tf.FixedLenSequenceFeature([], dtype=tf.string)}

contexts, features = tf.parse_single_sequence_example(
  serialized_examples, context_features=context_features, sequence_features=sequence_features)

decoded_rgb = tf.reshape(tf.cast(tf.decode_raw(features['rgb'], tf.uint8), tf.float32),
  shape=[-1, 1024])
decoded_audio = tf.reshape(tf.cast(tf.decode_raw(features['audio'], tf.uint8), tf.float32),
  shape=[-1, 128])

import cPickle as pkl
cluster_dir = '/data1/yj/yt8m_cluster/feature.pkl'

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  rgbs = []
  '''
  batch_size = 5
  vid_batch = tf.train.batch_join(
      [[decoded_rgb]],
      batch_size=batch_size,
      capacity=batch_size * 2,
      dynamic_pad=True)
  '''
  while not coord.should_stop():
    #print sess.run(features)
    '''
    print type(sess.run(features))
    print len(sess.run(features))
    print sess.run(features).keys()
    print type(sess.run(features)['rgb'])
    print sess.run(features)['rgb'].shape
    print sess.run(features)['audio'].shape
    print sess.run(contexts)
    print sess.run(contexts)['video_id']
    print sess.run(decoded_rgb).shape
    print sess.run(decoded_audio).shape
    '''
    drgb = sess.run(decoded_rgb)
    drgb = drgb[::20]
    rgbs.append(utils.Dequantize(drgb))
    if len(rgbs) % 10000 == 0:
        print 'doing...'
    if len(rgbs) > 80000:
        rgb_stack = np.concatenate(rgbs,axis=0)
        pkl.dump(rgb_stack,open(cluster_dir,'w'))
        break
  pudb.set_trace()
