from VLAD import *
import argparse
import glob
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils


'''
#parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--descriptorsPath", required = True,
	help = "Path to the file that contains the descriptors")
ap.add_argument("-w", "--numberOfVisualWords", required = True,
	help = "number of visual words or clusters to be computed")
ap.add_argument("-o", "--output", required = True,
	help = "Path to where the computed visualDictionary will be stored")
args = vars(ap.parse_args())
'''
#args
path = "/data1/yj/yt8m_cluster/audio_feature.pkl"
k = 256
#output
output="/data1/yj/yt8m_cluster/audio_cluster.pkl"


#computing the visual dictionary
print("estimating a visual dictionary of size: "+str(k)+ " for descriptors in path:"+path)
import cPickle as pkl

with open(path, 'rb') as f:
    descriptors=pkl.load(f)
#descriptors = np.zeros([20000,1024],dtype=np.float32)

def input_fn():
    return tf.constant(descriptors, dtype=tf.float32), tf.constant(descriptors,dtype=tf.float32)
#visualDictionary=kMeansDictionary_tf(descriptors,k)

est = tf.contrib.learn.KMeansClustering(
    num_clusters = k,
    relative_tolerance=0.0001,
    model_dir='/data1/yj/audio_kmeans/')


part_len = len(descriptors)/10 -1
#descriptors = descriptors[:part_len]

for n in range(10):
    for e in range(10):
        training = descriptors[e*part_len:(e+1)*part_len]
        for i in range(10):
            print 'epoch ' + str(e)
            print str(i) + 'iteration'
            _ = est.fit(input_fn=lambda: (constant_op.constant(training), None))





def serving_input_fn():
    serialized_tf_example = array_ops.placeholder(dtype=dtypes.string,
                                                shape=[None],
                                                name='input_example_tensor')
    features, labels = input_fn()
    return input_fn_utils.InputFnOps(
        features, labels, {'examples': serialized_tf_example})


#visualDictionary.export_savedmodel(export_dir_base='/data1/yj/kmeans',serving_input_fn=serving_input_fn)


#with open(output, 'wb') as f:
#	pickle.dump(visualDictionary, f)
import ipdb
ipdb.set_trace()
