import argparse
import math
import numpy as np
import socket
import importlib
import os
import sys
import time
import multiprocessing as mp
from multiprocessing import Pool
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from config_reg import *

# ------------------------------------------------------------------
# HOW TO USE
#  1. set which model to use
#  2. edit the evaluate() to fit in the model
#  3. assure the get_latent_vectors() will generate correct vectors for the model
#  4. run the python file in the console with options
#   eg.  python get_model_vectors.py --dataset_prefix business
#
# OPTIONS in console:
#  [necessary] --dataset_prefix business
#  [optional] --results_path ./results


# ------------------------------------------------------------------
# [constants]
# path to import
sys.path.append('..')
import models.PCAN as PNV
# the model's name
MODEL_NAME = 'pcan'
# the model file to resume
MODEL_FILE = '../MODEL_FILES/pcan.pth'
# where to output vectors
VECTOR_FOLDER = '../MODEL_VECTORS/' + MODEL_NAME

# other variables
EVAL_BATCH_SIZE = 2

# ------------------------------------------------------------------
# [variables that are set when running]

DATASET_PREFIX = None
DATABASE_VECTORS = []
QUERY_VECTORS = []
DATABASE_SETS = []
QUERY_SETS = []


def init(FLAGS):
    global DATASET_PREFIX
    DATASET_PREFIX = FLAGS.dataset_prefix

    global DATABASE_SETS, QUERY_SETS
    EVAL_DATABASE_FILE = get_dataset_pickle_file(FLAGS.dataset_prefix)
    DATABASE_SETS = get_sets_dict(EVAL_DATABASE_FILE)
    EVAL_QUERY_FILE = get_dataset_pickle_file(FLAGS.dataset_prefix, type='query')
    QUERY_SETS = get_sets_dict(EVAL_QUERY_FILE)


def evaluate():
    model = PNV.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
                             output_dim=256, num_points=4096)
    model = model.to(device)

    resume_filename = MODEL_FILE
    print("Resuming From ", resume_filename)
    checkpoint = torch.load(resume_filename)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    model = nn.DataParallel(model)

    evaluate_model(model)


def evaluate_model(model):
    # ---------------------------------------------------------
    path_save = os.path.join(VECTOR_FOLDER)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    print(f"Vectors will be saved to {path_save}")
    print("Vectors calculating start...")

    for i in range(len(DATABASE_SETS)):
        DATABASE_VECTORS.append(get_latent_vectors(model, DATABASE_SETS[i]))

    for j in range(len(QUERY_SETS)):
        QUERY_VECTORS.append(get_latent_vectors(model, QUERY_SETS[j]))

    print("Vectors calculating done.")
    filename = f'{DATASET_PREFIX}_dataset_vectors.pickle'
    f = open(os.path.join(path_save, filename), 'wb')
    pickle.dump(DATABASE_VECTORS, f)
    f.close()
    print(f"Save {filename} success.")

    filename = f'{DATASET_PREFIX}_query_vectors.pickle'
    f = open(os.path.join(path_save, filename), 'wb')
    pickle.dump(QUERY_VECTORS, f)
    f.close()
    print(f"Save {filename} success.")
    # ---------------------------------------------------------


def get_latent_vectors(model, dict_to_process):
    model.eval()
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = EVAL_BATCH_SIZE * \
                (1 + FLAGS.positives_per_query + FLAGS.negatives_per_query)
    q_output = []
    for q_index in range(len(train_file_idxs) // batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index + 1) * (batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pts_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            out, weights = model(feed_tensor)
        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        # out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if (len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pts_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            o1, weights = model(feed_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    return q_output


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--positives_per_query', type=int, default=4,
                        help='Number of potential positives in each training tuple [default: 2]')
    parser.add_argument('--negatives_per_query', type=int, default=12,
                        help='Number of definite negatives in each training tuple [default: 20]')
    add_params_for_parser(parser)

    FLAGS = parser.parse_args()
    print(FLAGS)
    init(FLAGS)

    evaluate()
