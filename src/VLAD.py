import numpy as np
import itertools
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import pickle
import glob
import cv2
import tensorflow as tf

def  kMeansDictionary(training, k, max_iter=290):

    #K-means algorithm
    est = KMeans(n_clusters=k,init='k-means++',tol=0.0001,verbose=1).fit(training)
    #centers = est.cluster_centers_
    #labels = est.labels_
    #est.predict(X)
    return est



def  kMeansDictionary_tf(training, k, max_iter=290, input_fn=None):

    #K-means algorithm
    #est = KMeans(n_clusters=k,init='k-means++',tol=0.0001,verbose=1).fit(training)
    import ipdb
    ipdb.set_trace()
    est = tf.contrib.learn.KMeansClustering(
        num_clusters=k, relative_tolerance=0.0001)
    _ = est.fit(input_fn=input_fn)
    #centers = est.cluster_centers_
    #labels = est.labels_
    #est.predict(X)
    return est


def VLAD_tf(X,visualDictionary):

    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.clusters()
    labels=visualDictionary.labels_
    k=visualDictionary.num_clusters

    m,d = X.shape
    V=tf.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i]=tf.sum(X[predictedLabels==i,:]-centers[i],axis=0)


    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = tf.sign(V)*tf.sqrt(tf.abs(V))

    # L2 normalization

    V = V/tf.sqrt(tf.tensordot(V,V))
    return V



def VLAD(X,visualDictionary):

    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels=visualDictionary.labels_
    k=visualDictionary.n_clusters

    m,d = X.shape
    V=np.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)


    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization

    V = V/np.sqrt(np.dot(V,V))
    return V


def improvedVLAD(X,visualDictionary):

    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels=visualDictionary.labels_
    k=visualDictionary.n_clusters

    m,d = X.shape
    V=np.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)


    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization

    V = V/np.sqrt(np.dot(V,V))
    return V

def indexBallTree(X,leafSize):
    tree = BallTree(X, leaf_size=leafSize)
    return tree



