import argparse
import math
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import tensorflow as tf
import socket
import importlib
import sys
import copy
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('..')
from pointnetvlad_cls import *
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

import time
import pickle

from config_reg import *

# ------------------------------------------------------------------
# [constants]# the model's fullpath
MODEL_FILE = '../models/baseline/model_baseline.ckpt'
# the model's name
MODEL_NAME = 'pointnetvlad'
# where to output vectors
VECTOR_FOLDER = '../MODEL_VECTORS/' + MODEL_NAME

# ------------------------------------------------------------------
# [variables that are set when running]
DATASET_PREFIX = None

DATABASE_VECTORS = []
QUERY_VECTORS = []
DATABASE_SETS = []
QUERY_SETS = []

DECAY_STEP = None
DECAY_RATE = None


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_NUM_QUERIES,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def evaluate():
    global DATABASE_VECTORS
    global QUERY_VECTORS

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            print("In Graph")
            query= placeholder_inputs(BATCH_NUM_QUERIES, 1, NUM_POINTS)
            positives= placeholder_inputs(BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS)
            negatives= placeholder_inputs(BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS)
            eval_queries= placeholder_inputs(EVAL_BATCH_SIZE, 1, NUM_POINTS)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)

            with tf.variable_scope("query_triplets") as scope:
                vecs= tf.concat([query, positives, negatives],1)
                out_vecs= forward(vecs, is_training_pl, bn_decay=bn_decay)
                q_vec, pos_vecs, neg_vecs= tf.split(out_vecs, [1,POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY],1)

            saver = tf.train.Saver()

        # Create a session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)


        saver.restore(sess, MODEL_FILE)
        print("Model restored.")

        ops = {'query': query,
               'positives': positives,
               'negatives': negatives,
               'is_training_pl': is_training_pl,
               'eval_queries': eval_queries,
               'q_vec':q_vec,
               'pos_vecs': pos_vecs,
               'neg_vecs': neg_vecs}
        recall= np.zeros(25)
        count=0
        similarity=[]
        one_percent_recall=[]
        for i in range(len(DATABASE_SETS)):
            DATABASE_VECTORS.append(get_latent_vectors(sess, ops, DATABASE_SETS[i]))

        for j in range(len(QUERY_SETS)):
            QUERY_VECTORS.append(get_latent_vectors(sess, ops, QUERY_SETS[j]))

        print('m x n', len(QUERY_SETS) * len(QUERY_SETS))

    # ---------------------------------------------------------
    # only output model's results
    path_save = os.path.join(VECTOR_FOLDER)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    print(f"Vectors will be saved to {path_save}")

    filename = f'{DATASET_PREFIX}_dataset_vectors.pickle'
    f = open(os.path.join(path_save, filename), 'wb')
    pickle.dump(DATABASE_VECTORS, f)
    f.close()
    print(f"Save {filename} file success.")

    filename = f'{DATASET_PREFIX}_query_vectors.pickle'
    f = open(os.path.join(path_save, filename), 'wb')
    pickle.dump(QUERY_VECTORS, f)
    f.close()
    print(f"Save {filename} file success.")
    # ---------------------------------------------------------
    return


def get_latent_vectors(sess, ops, dict_to_process):
    is_training=False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))
    #print(len(train_file_idxs))
    batch_num= BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY)
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices=train_file_idxs[q_index*batch_num:(q_index+1)*(batch_num)]
        file_names=[]
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries=load_pts_files(file_names)
        # queries= np.expand_dims(queries,axis=1)
        q1=queries[0:BATCH_NUM_QUERIES]
        q1=np.expand_dims(q1,axis=1)
        #print(q1.shape)

        q2=queries[BATCH_NUM_QUERIES:BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1)]
        q2=np.reshape(q2,(BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))

        q3=queries[BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1):BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1)]
        q3=np.reshape(q3,(BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))
        feed_dict={ops['query']:q1, ops['positives']:q2, ops['negatives']:q3, ops['is_training_pl']:is_training}
        o1, o2, o3=sess.run([ops['q_vec'], ops['pos_vecs'], ops['neg_vecs']], feed_dict=feed_dict)

        o1=np.reshape(o1,(-1,o1.shape[-1]))
        o2=np.reshape(o2,(-1,o2.shape[-1]))
        o3=np.reshape(o3,(-1,o3.shape[-1]))

        out=np.vstack((o1,o2,o3))
        q_output.append(out)

    q_output=np.array(q_output)
    if(len(q_output)!=0):
        q_output=q_output.reshape(-1,q_output.shape[-1])
    #print(q_output.shape)

    #handle edge case
    for q_index in range((len(train_file_idxs)//batch_num*batch_num),len(dict_to_process.keys())):
        index=train_file_idxs[q_index]
        queries=load_pts_files([dict_to_process[index]["query"]])
        queries= np.expand_dims(queries,axis=1)
        #print(query.shape)
        #exit()
        fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
        fake_pos=np.zeros((BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))
        fake_neg=np.zeros((BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))
        q=np.vstack((queries,fake_queries))
        #print(q.shape)
        feed_dict={ops['query']:q, ops['positives']:fake_pos, ops['negatives']:fake_neg, ops['is_training_pl']:is_training}
        output=sess.run(ops['q_vec'], feed_dict=feed_dict)
        #print(output.shape)
        output=output[0]
        output=np.squeeze(output)
        if (q_output.shape[0]!=0):
            q_output=np.vstack((q_output,output))
        else:
            q_output=output

    #q_output=np.array(q_output)
    #q_output=q_output.reshape(-1,q_output.shape[-1])
    #print(q_output.shape)
    return q_output


if __name__ == "__main__":

    #params
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--positives_per_query', type=int, default=2, help='Number of potential positives in each training tuple [default: 2]')
    parser.add_argument('--negatives_per_query', type=int, default=18, help='Number of definite negatives in each training tuple [default: 20]')
    parser.add_argument('--batch_num_queries', type=int, default=1, help='Batch Size during training [default: 1]')
    parser.add_argument('--dimension', type=int, default=256)
    parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')

    add_params_for_parser(parser)

    FLAGS = parser.parse_args()
    print(FLAGS)

    #BATCH_SIZE = FLAGS.batch_size
    BATCH_NUM_QUERIES = FLAGS.batch_num_queries
    EVAL_BATCH_SIZE = 1
    NUM_POINTS = 4096
    POSITIVES_PER_QUERY= FLAGS.positives_per_query
    NEGATIVES_PER_QUERY= FLAGS.negatives_per_query
    GPU_INDEX = FLAGS.gpu
    DECAY_STEP = FLAGS.decay_step
    DECAY_RATE = FLAGS.decay_rate

    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_DECAY_STEP = float(DECAY_STEP)
    BN_DECAY_CLIP = 0.99

    DATABASE_FILE= get_dataset_pickle_file(FLAGS.dataset_prefix)
    QUERY_FILE= get_dataset_pickle_file(FLAGS.dataset_prefix, type='query')

    DATABASE_SETS= get_sets_dict(DATABASE_FILE)
    QUERY_SETS= get_sets_dict(QUERY_FILE)

    DATASET_PREFIX = FLAGS.dataset_prefix

    evaluate()
