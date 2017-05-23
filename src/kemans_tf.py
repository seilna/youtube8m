
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans

# pylint: disable=g-import-not-at-top
from tensorflow.contrib.learn.python import learn
from tensorflow.contrib.learn.python.learn.estimators import kmeans as kmeans_lib
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import flags
from tensorflow.python.platform import test
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import queue_runner

FLAGS = flags.FLAGS

import pudb


def normalize(x):
  return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True))


def cosine_similarity(x, y):
  return np.dot(normalize(x), np.transpose(normalize(y)))


def make_random_centers(num_centers, num_dims, center_norm=500):
  return np.round(
      np.random.rand(num_centers, num_dims).astype(np.float32) * center_norm)


def make_random_points(centers, num_points, max_offset=20):
  num_centers, num_dims = centers.shape
  assignments = np.random.choice(num_centers, num_points)
  offsets = np.round(
      np.random.randn(num_points, num_dims).astype(np.float32) * max_offset)
  return (centers[assignments] + offsets, assignments,
          np.add.reduce(offsets * offsets, 1))


class KMeansTestBase(test.TestCase):

  def input_fn(self, batch_size=None, points=None, randomize=None,
               num_epochs=None):
    """Returns an input_fn that randomly selects batches from given points."""
    batch_size = batch_size or self.batch_size
    points = points if points is not None else self.points
    num_points = points.shape[0]
    if randomize is None:
      randomize = (self.use_mini_batch and
                   self.mini_batch_steps_per_iteration <= 1)
    def _fn():
      x = constant_op.constant(points)
      if batch_size == num_points:
        return input_lib.limit_epochs(x, num_epochs=num_epochs), None
      if randomize:
        indices = random_ops.random_uniform(
            constant_op.constant([batch_size]),
            minval=0, maxval=num_points-1,
            dtype=dtypes.int32,
            seed=10)
      else:
        # We need to cycle through the indices sequentially. We create a queue
        # to maintain the list of indices.
        q = data_flow_ops.FIFOQueue(self.num_points, dtypes.int32, ())
        # Conditionally initialize the Queue.
        def _init_q():
          with ops.control_dependencies([q.enqueue_many(
              math_ops.range(self.num_points))]):
            return control_flow_ops.no_op()
        init_q = control_flow_ops.cond(q.size() <= 0,
                                       _init_q,
                                       control_flow_ops.no_op)
        with ops.control_dependencies([init_q]):
          offsets = q.dequeue_many(self.batch_size)
          with ops.control_dependencies([q.enqueue_many(offsets)]):
            indices = array_ops.identity(offsets)
      batch = array_ops.gather(x, indices)
      return (input_lib.limit_epochs(batch, num_epochs=num_epochs), None)
    return _fn

  @staticmethod
  def config(tf_random_seed):
    return run_config.RunConfig(tf_random_seed=tf_random_seed)

  @property
  def batch_size(self):
    return self.num_points

  @property
  def use_mini_batch(self):
    return False

  @property
  def mini_batch_steps_per_iteration(self):
    return 1


class KMeansTest(KMeansTestBase):

  def setUp(self):
    np.random.seed(3)
    self.num_centers = 5
    self.num_dims = 2
    self.num_points = 1000
    self.points = np.zeros([10000,1024])

  def _kmeans(self, relative_tolerance=None):
    return kmeans_lib.KMeansClustering(
        self.num_centers,
        initial_clusters=kmeans_lib.KMeansClustering.KMEANS_PLUS_PLUS_INIT,
        distance_metric=kmeans_lib.KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE,
        use_mini_batch=self.use_mini_batch,
        mini_batch_steps_per_iteration=self.mini_batch_steps_per_iteration,
        random_seed=24,
        relative_tolerance=relative_tolerance)

  def test_clusters(self):
    kmeans = self._kmeans()
    kmeans.fit(input_fn=self.input_fn(), steps=1)
    clusters = kmeans.clusters()
    self.assertAllEqual(list(clusters.shape), [self.num_centers, self.num_dims])

  def test_fit(self):
    kmeans = self._kmeans()
    kmeans.fit(input_fn=self.input_fn(), steps=1)
    score1 = kmeans.score(
        input_fn=self.input_fn(batch_size=self.num_points), steps=1)
    steps = 10 * self.num_points // self.batch_size
    kmeans.fit(input_fn=self.input_fn(), steps=steps)
    score2 = kmeans.score(
        input_fn=self.input_fn(batch_size=self.num_points), steps=1)
    self.assertTrue(score1 > score2)

  def test_monitor(self):
    if self.use_mini_batch:
      # We don't test for use_mini_batch case since the loss value can be noisy.
      return
    kmeans = kmeans_lib.KMeansClustering(
        self.num_centers,
        initial_clusters=kmeans_lib.KMeansClustering.KMEANS_PLUS_PLUS_INIT,
        distance_metric=kmeans_lib.KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE,
        use_mini_batch=self.use_mini_batch,
        mini_batch_steps_per_iteration=self.mini_batch_steps_per_iteration,
        config=learn.RunConfig(tf_random_seed=14),
        random_seed=12,
        relative_tolerance=1e-4)

    kmeans.fit(
        input_fn=self.input_fn(),
        # Force it to train until the relative tolerance monitor stops it.
        steps=None)
    score = kmeans.score(
        input_fn=self.input_fn(batch_size=self.num_points), steps=1)


pudb.set_trace()
