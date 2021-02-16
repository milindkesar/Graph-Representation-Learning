import tensorflow as tf
from graphsage.utils import load_data
from graphsage.minibatch import WeightedNodeMinibatchIterator
import numpy as np
from sklearn.externals import joblib
import networkx as nx
def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'), # dynamic
        'batch' : tf.placeholder(tf.int32, shape=(None), name='batch1'), # dynamic
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'), # scalar
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders
G, feats, id_map, walks, class_map = load_data('/home/thummala/graph-datasets/ppi/ppi')
centroids=np.load('/home/thummala/graph-datasets/ppi_extra/centroids.npy')
CModel=joblib.load('/home/thummala/graph-datasets/ppi_extra/ppi.skl')
if isinstance(list(class_map.values())[0], list):
    num_classes = len(list(class_map.values())[0])
else:
    num_classes = len(set(class_map.values()))
placeholders=construct_placeholders(num_classes)
minibatch = WeightedNodeMinibatchIterator(CModel,
                                          centroids,
                                          dif_weight=1,
                                          directed_graph=False,
                                          # sampling 128 neighbors randomly considering directed condition
                                          G=G,
                                          id2idx=id_map,
                                          placeholders=placeholders,
                                          label_map=class_map,
                                          num_classes=num_classes,
                                          batch_size=256,
                                          max_degree=128,
                                          context_pairs=None)
adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
adj_weight_ph = tf.placeholder(tf.float32, shape=minibatch.adj.shape)
adj_weight = tf.Variable(adj_weight_ph, trainable=False, name="adj_weight")
print(adj_weight.shape) #Nodes + 1 * Max Degree Tensor
print(adj_info.shape)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
#print(sess.run(sess.global_varia[adj_info[-1]]),feed_dict={adj_info_ph:adj_info,adj_weight_ph:adj_weight})
count=0
while count<3:
    feed_dict,labels = minibatch.next_minibatch_feed_dict()
    feed_dict.update({placeholders['dropout']:0})
    print(sess.run([adj_weight]),feed_dict=feed_dict)
    count=count+1
sess.close()
#print(G.nodes())# 56944 nodes are there
# print(nx.info(G))
# print("Size of node features :" +str(feats.shape[1]))
# if isinstance(list(class_map.values())[0], list):
#     num_classes = len(list(class_map.values())[0])
# else:
#     num_classes = len(set(class_map.values()))
# print('Number of classes is '+str(num_classes))
# print(len(class_map[0]))
# print(len(class_map.keys()))
# print('Features shape : '+str(feats.shape))
