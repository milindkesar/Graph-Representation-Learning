from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists

## My changes

class WeightedNeighborSampler(Layer):
    """
    samples neighbors with bias.
    """
    def __init__(self, adj_info, adj_weight, **kwargs):
        super(WeightedNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.adj_weight = adj_weight

    def _call(self, inputs):
        ids, num_samples = inputs

        with tf.name_scope('weighted_sampler') as scope:
            adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
            print('adj list '+str(adj_lists.shape))
            adj_weight_lists = tf.nn.embedding_lookup(self.adj_weight, ids)
            idx = tf.multinomial(tf.log(adj_weight_lists), num_samples)
            bias = tf.expand_dims(tf.range(tf.shape(idx, out_type=tf.int64)[0]), 1)
            print(bias.shape)
            expand_idx = tf.reshape(bias * tf.shape(adj_lists, out_type=tf.int64)[1] + idx, [-1])
            res1=tf.gather(tf.reshape(adj_lists,[-1]),expand_idx)
            res=tf.reshape(res1,[-1,num_samples])
            #res = tf.reshape(tf.gather(tf.reshape(adj_lists,[-1]), expand_idx), (-1, num_samples))
        return res


