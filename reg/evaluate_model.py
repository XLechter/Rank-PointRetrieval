import argparse
import numpy as np
import os
import sys
import copy
import time
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import multiprocessing as mp
from multiprocessing import Pool

from config_reg import *
import get_regist_score
import draw_reg_result
import tracemalloc

# ------------------------------------------------------------------
# HOW TO USE
#  1. assure vectors exsit for the model to use
#     check [constants] for where will be checked for vectors
#  2. assure the dataset's path in config_reg.py is correct
#  3. run the python file in the console with options
#   eg.  python evaluate_model.py --model pcan --dataset_prefix business
#     (the vectors is in VECTOR_FOLDER/pcan/)
#
# OPTIONS in console:
#  [necessary] --model pcan
#  [necessary] --dataset_prefix business
#  [optional] --results_path ./results
#  others could be found by using --help


# ------------------------------------------------------------------
# [constants]
# which path to load the model vector file
#  if using the model 'pcan', the dataset prefix 'university',
#  then the vectors files(got from get_model_vectors.py) must be
#    VECTOR_FOLDER/pcan/university_dataset_vectors.pickle and
#    VECTOR_FOLDER/pcan/university_query_vectors.pickle
# the DATA_FOLDER is set in config_reg.py, where to store data for accelerating
VECTOR_FOLDER = '../MODEL_VECTORS'
# numbers that be selected to do re-ranking
num_match = 8

# ------------------------------------------------------------------
# [constants that are set from console]
# using reg to do re-ranking
with_REG = True
# output logs
OUTPUT_LOG = False
OUTPUT_LOG_PLY = False
# the dataset that will be used to evaluate
DATASET_PREFIX = None
# global weights
lamda1 = None
lamda2 = None
# the txt file that stores outputs
OUTPUT_FILE = None

# ------------------------------------------------------------------
# [variables that are set when running]
# the start running time
TIME_START = time.strftime('%m.%d %H:%M')

sum_recall = mp.Array("f", [0] * num_match)
sum_count = mp.Value('i', 0)
sum_similarity = mp.Manager().list()
sum_one_percent_recall = mp.Manager().list()

DATABASE_VECTORS = []
QUERY_VECTORS = []
DATABASE_SETS = []
QUERY_SETS = []


def init(FLAGS):
    get_regist_score.init(
      dataset_prefix=FLAGS.dataset_prefix,
      reg_type = FLAGS.reg,
      with_ppf_feature=FLAGS.with_ppf_feature,
      keypoints=FLAGS.keypoints,
      keytype=FLAGS.keytype,
      with_noise=FLAGS.with_noise,
      sigma=FLAGS.sigma
    )

    global DATABASE_SETS, QUERY_SETS, DATABASE_VECTORS, QUERY_VECTORS
    # read the database
    DATABASE_FILE = get_dataset_pickle_file(FLAGS.dataset_prefix)
    DATABASE_SETS = get_sets_dict(DATABASE_FILE)
    # read the query file
    QUERY_FILE = get_dataset_pickle_file(FLAGS.dataset_prefix, type='query')
    QUERY_SETS = get_sets_dict(QUERY_FILE)
    # read the vectors calculated from any model
    try:
        f1_name = f'{DATASET_PREFIX}_dataset_vectors.pickle'
        f2_name = f'{DATASET_PREFIX}_query_vectors.pickle'

        f1_fullfile = os.path.join(VECTOR_FOLDER, FLAGS.model, f1_name)
        f2_fullfile = os.path.join(VECTOR_FOLDER, FLAGS.model, f2_name)

        f1 = open(f1_fullfile, 'rb')
        f2 = open(f2_fullfile, 'rb')

        DATABASE_VECTORS = pickle.load(f1)
        print(f"Load vectors from {f1_fullfile} success.")

        QUERY_VECTORS = pickle.load(f2)
        print(f"Load vectors from {f2_fullfile} success.")

    except Exception as ex:
        print(f"[ERROR] {type(ex)} | {ex.args!r}")
        print(f"Load {FLAGS.model}'s vectors files failed!")
        return


def evaluate():
    global sum_recall
    global sum_count
    global sum_similarity
    global sum_one_percent_recall

    start = time.time()
    index = []
    for m in range(len(QUERY_SETS)):
        for n in range(len(QUERY_SETS)):
            if n != m:
                index.append([m, n])

    cpu_count = mp.cpu_count()
    #cpu_using = int(cpu_count / 2)
    cpu_using = 32
    print(f"cpu count: {cpu_count} | cpu using: {cpu_using}")
    p = Pool(cpu_using)
    p.map(get_recall, index)
    p.close()
    ave_recall = np.array(sum_recall) / sum_count.value
    average_similarity = np.mean(sum_similarity)
    ave_one_percent_recall = np.mean(sum_one_percent_recall)
    print("Average Top 1% Recall: ", ave_one_percent_recall)
    print("TIme using: %.3f sec" % (time.time() - start))

    with open(OUTPUT_FILE, "w") as output:
        output.write("Average Recall @N:\n")
        output.write(str(ave_recall))
        output.write("\n\n")
        output.write("Average Similarity:\n")
        output.write(str(average_similarity))
        output.write("\n\n")
        output.write("Average Top 1% Recall:\n")
        output.write(str(ave_one_percent_recall))
        output.write("\n\n")

        output.write(" Start Time: " + TIME_START + "\n")
        output.write("Finish Time: " + time.strftime('%m.%d %H:%M') + "\n")
        output.write("      Using: %.3f sec\n"%(time.time() - start))

    print(f"Results have been saved to {OUTPUT_FILE} successfully.")


def get_recall(index):
    global sum_recall
    global sum_count
    global sum_similarity
    global sum_one_percent_recall
    m = index[0]
    n = index[1]

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    #print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    #print('database_output.shape: ', database_output.shape)
    database_nbrs = KDTree(database_output)

    recall = [0] * num_match

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        true_neighbors = QUERY_SETS[n][i][m]
        if (len(true_neighbors) == 0):
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]), k=num_match)

        #print("---------------------------------------")
        print("current: m%d - n%d - i%d"%(m, n, i))
        #print("    gts: ", end="")
        #show_matchs([], true_neighbors)
        #print(" matchs: ", end="")
        #show_matchs(true_neighbors, indices[0][0:num_match])

        results_reg = []  # storing results after re-ranking
        if with_REG == True:  # re-rank with reg results
            files_match = []
            for i_mt in indices[0][0:num_match]:
                file_mt = {}
                file_mt['file'] = DATABASE_SETS[m][i_mt]['query']  # get match's filename
                file_mt['id2'] = i_mt
                file_mt['gt'] = i_mt in true_neighbors
                file_mt['cur'] = "m%dn%di%d"%(m, n, i)
                files_match.append(file_mt)
            file_cur = QUERY_SETS[n][i]['query']  # get current's filename


            #print('file cure match: ', file_cur, files_match)
            tracemalloc.start()
            rs = get_regist_score.evaluate_matchs(file_cur, files_match, log=False)
            print(tracemalloc.get_traced_memory())
            tracemalloc.stop()

            match_score_tmp = []
            for _i in range(len(rs)):
                score1 = 1.0 / distances[0][_i]
                score2 = rs[_i]['reg_score']
                score = score1 * lamda1 + score2 * lamda2

                match_score_tmp.append([indices[0][_i], score1, score2, score, rs[_i]])

            # save the init scores for outputting logs
            match_score_init = copy.deepcopy(match_score_tmp)

            def takeScore(elem):
                return elem[3]

            match_score_tmp.sort(reverse=True, key=takeScore)
            results_reg = [_i[0] for _i in match_score_tmp]
            #print("results: ", end="")
            #show_matchs(true_neighbors, results_reg)

            # ----------------------------------------------------------
            # output logs
            # ----------------------------------------------------------
            if OUTPUT_LOG and (indices[0][0] not in true_neighbors or results_reg[0] not in true_neighbors):
                PATH_CURRENT = OUTPUT_FILE[0:-4] + "/m%d_n%d_i%d"%(m, n, i)
                if not os.path.exists(PATH_CURRENT):
                    os.makedirs(PATH_CURRENT)
                # current pointcloud
                _str_cur = QUERY_SETS[n][i]['query']
                if OUTPUT_LOG_PLY:
                    # output current pcd's ply
                    draw_reg_result.draw_ply_from_file(PATH_CURRENT+"/cur.ply", _str_cur, 's')
                    # output current pcd's keypoints' ply
                    if get_regist_score.num_keypts > 0:
                        kids = get_regist_score.data_kids[_str_cur]["kids"]
                        draw_reg_result.draw_keypoints_from_file(PATH_CURRENT+"/cur_keypts.ply", _str_cur, kids)
                # output gts
                f_log = open(PATH_CURRENT + '/log_gt.txt', 'w')
                f_log.write('GT match for %d: '%i + _str_cur)
                for _i, gt in enumerate(true_neighbors):
                    _str = DATABASE_SETS[m][gt]['query']
                    f_log.write('\n <gt>%3d <file>' % gt + _str)
                    # output gt's ply
                    if OUTPUT_LOG_PLY:
                        draw_reg_result.draw_ply_from_file(PATH_CURRENT+"/gt%d_%d.ply"%(_i+1, gt), _str)
                f_log.close()
                # output mts
                f_log = open(PATH_CURRENT + '/log_mt.txt', 'w')
                f_log.write('Init match for %d: '%i + _str_cur)
                for _i, mt in enumerate(indices[0]):
                    _str = DATABASE_SETS[m][mt]['query']
                    flag = ' '
                    if mt in true_neighbors:
                        flag = '*'
                    f_log.write('\n %s <mt>%3d <score> mt:%f <file>' % (flag, mt, match_score_init[_i][1]) + _str)
                #   find the last gt in mts, and output 0-to-it's plys
                if OUTPUT_LOG_PLY:
                    _i_end = 0
                    for _i in range(0, num_match):
                        if indices[0][_i] in true_neighbors:
                            _i_end = _i
                    if _i_end < 4: # make sure minimum ply
                        _i_end = 4
                    for _i, mt in enumerate(indices[0][0:(_i_end+1)]):
                        _str = DATABASE_SETS[m][mt]['query']
                        draw_reg_result.draw_ply_from_file(PATH_CURRENT+"/mt%d_%d.ply"%(_i+1, mt), _str)
                        draw_reg_result.draw_reg_from_file(PATH_CURRENT+"/mt%d_%d-cur.ply"%(_i+1, mt), _str_cur, _str, match_score_init[_i][-1]['trans'])
                f_log.close()
                # output reg results
                f_log = open(PATH_CURRENT + '/log_rg.txt', 'w')
                f_log.write('After reg for %d: '%i + _str_cur)
                for _i, rg in enumerate(results_reg):
                    _str = DATABASE_SETS[m][rg]['query']
                    flag = ' '
                    if rg in true_neighbors:
                        flag = '*'
                    f_log.write('\n %s <rg>%3d <score> mt:%f rg:%f s:%f <file>' % (flag, rg, match_score_tmp[_i][1], match_score_tmp[_i][2], match_score_tmp[_i][3]) + _str)
                #   find the last gt in rgs, output 0-to-it's plys
                if OUTPUT_LOG_PLY:
                    _i_end = 0
                    for _i in range(0, num_match):
                        if results_reg[_i] in true_neighbors:
                            _i_end = _i
                    if _i_end < 4: # make sure minimum ply
                        _i_end = 4
                    for _i, rg in enumerate(results_reg[0:(_i_end+1)]):
                        _str = DATABASE_SETS[m][rg]['query']
                        draw_reg_result.draw_ply_from_file(PATH_CURRENT+"/rg%d_%d.ply"%(_i+1, rg), _str)
                        draw_reg_result.draw_reg_from_file(PATH_CURRENT+"/rg%d_%d-cur.ply"%(_i+1, rg), _str_cur, _str, match_score_tmp[_i][-1]['trans'])
                f_log.close()
            # ----------------------------------------------------------

        else:
            results_reg = indices[0]

        for j in range(len(results_reg)):
            if results_reg[j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(
                        queries_output[i], database_output[results_reg[0]])
                    top1_similarity_score.append(similarity)
                # probability that a GT is at j
                recall[j] += 1
                break

        if len(list(set(results_reg[0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100

    sum_recall[:] = sum_recall[:] + np.array(recall)
    sum_count.value += 1
    sum_one_percent_recall.append(one_percent_recall)

    for x in top1_similarity_score:
        sum_similarity.append(x)

    return


def show_matchs(gts, matchs):
    arr = [str(i) for i in matchs]
    for i, v in enumerate(matchs):
        if v in gts:
            arr[i] += "[√]"
    print(" ".join(arr))


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    add_params_for_parser(parser)

    # to set which model's data to use
    parser.add_argument('--model', default='pointnetvlad',
                        help='Using which model\'s result to do reg, the data of dataset_vectors and query_vectors must be under DATA_FOLDER/{model} [default: pointnetvlad]')

    FLAGS = parser.parse_args()
    FLAGS.multi = True
    print(FLAGS)

    DATASET_PREFIX = FLAGS.dataset_prefix
    with_REG = FLAGS.with_reg
    OUTPUT_LOG = FLAGS.log
    OUTPUT_LOG_PLY = FLAGS.log_ply
    lamda1 = FLAGS.lamda1
    lamda2 = FLAGS.lamda2
    init(FLAGS)

    # to distinct the result file
    FLAGS.surfix += f"_{FLAGS.model}"

    OUTPUT_FILE = get_result_txt_name(FLAGS)
    print("Will save to: ", OUTPUT_FILE)
    if OUTPUT_LOG:
        print("Logs will save to: ", OUTPUT_FILE[0:-4]+"/")
        if OUTPUT_LOG_PLY:
            print("Results of ply files will also save to this path.")

    evaluate()
