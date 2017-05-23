
import numpy as np
import tensorflow as tf
import sys
import time
import cPickle as pkl

num_clusters = 1024
num_steps = 500
print("Kmeans (Optimal Version) with " + str(num_clusters) + " clusters and " + str(num_steps) + " steps")
begin_io_time = time.time()
# Read input_file
#vector_values = pkl.load(open('/data1/yj/yt8m_cluster/feature.pkl'))
vector_values = np.zeros([10000,1024], dtype=np.float32)
# Delete first row (id)
#vector_values = np.delete(vector_values, 0, 1)

print("Total IO Time: %3.2fs" % float(time.time() - begin_io_time))

with tf.device('/gpu:0'):
    vectors = tf.constant(vector_values)
    centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),
                                     [0, 0], [num_clusters, -1]))
    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)

    distances = tf.reduce_sum(
        tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    assignments = tf.argmin(distances, 0)

    means = tf.concat([
        tf.reduce_mean(
            tf.gather(vectors,
                      tf.reshape(
                          tf.where(
                              tf.equal(assignments, c)
                          ), [1, -1])
                      ), reduction_indices=[1])
        for c in range(num_clusters)],axis=0)

    update_centroids = tf.assign(centroids, means)
    init_op = tf.initialize_all_variables()

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
sess.run(init_op)

begin_time = time.time()
for step in range(num_steps):
    _, centroid_values, assignment_values = sess.run([update_centroids,
                                                      centroids,
                                                      assignments])
print("Total Ex Time: %3.2fs" % float(time.time() - begin_time))

print ("Centroids: " + str(centroid_values))
