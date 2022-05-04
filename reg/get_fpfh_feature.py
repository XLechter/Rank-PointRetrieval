import sys
import numpy as np
import open3d
import os
import pickle
import copy
import time

import config_reg

PATH_SAVE_ROOT = config_reg.DATA_FOLDER + '/DATA_FEATURES_FPFH'
if not os.path.exists(PATH_SAVE_ROOT):
    os.makedirs(PATH_SAVE_ROOT)

# ------------------------------------------------------------------
def calc_fpfh_feature(s):
    radius_normal = 0.05
    start = time.time()
    s.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    print('normal time', time.time()-start)
    
    radius_feature = 0.05
    start = time.time()
    s_feature = open3d.registration.compute_fpfh_feature(s, open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    print('fpfh time', time.time()-start)
    # print( type(s_feature.data) ) # np.ndarray
    # s_fpfh.data is a (33, n) nd array, n is the points number
    return s_feature


# ------------------------------------------------------------------
# get a pointcloud's fpfh feature
# ------------------------------------------------------------------
#   filename : the pointcloud file in the DATASET_ROOT path
def save_fpfh_feature(filename, dataset_prefix, save=True):
    pts = config_reg.load_pts_file(filename)
    s = open3d.geometry.PointCloud()
    s.points = open3d.utility.Vector3dVector(pts)

    s_feature = calc_fpfh_feature(s)

    if save:
        dir_ = os.path.join(PATH_SAVE_ROOT, dataset_prefix, os.path.dirname(filename))
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        file_save = os.path.join(PATH_SAVE_ROOT, dataset_prefix, filename+".npz")
        np.savez(file_save, fpfh_f=s_feature.data)
        # s_feature.data.tofile(file_save)

    return s_feature


def load_fpfh_feature(filename, dataset_prefix):
    file_f = os.path.join(PATH_SAVE_ROOT, dataset_prefix, filename+".npz")
    if os.path.isfile(file_f):
        # fs = np.fromfile(file_f, dtype=np.float)
        r = np.load(file_f)

        s_feature = open3d.registration.Feature()
        s_feature.data = r['fpfh_f']

        return s_feature
    else:
        s_feature = save_fpfh_feature(filename, dataset_prefix)
        return s_feature


# ------------------------------------------------------------------
# get a dataset's fpfh
# ------------------------------------------------------------------
def get_feature_file(dataset_prefix):
    return PATH_SAVE_ROOT + "/" + dataset_prefix + ".pickle"


# save the dataset's fpfh features in one file
def save_fpfh_pickle(dataset_prefix, database_file=""):
    DATABASE_FILE = config_reg.get_dataset_pickle_file(dataset_prefix)
    if database_file != "":
        DATABASE_FILE = database_file

    try:
        DATABASE_SETS = config_reg.get_sets_dict(DATABASE_FILE)
    except Exception as ex:
        print("[ERROR] Load database failed! No such file: " + DATABASE_FILE)
        print(f"[ERROR] {type(ex)} | {ex.args!r}")
        return {}

    print("Processing FPFH feature: " + DATABASE_FILE)

    f_name = get_feature_file(dataset_prefix)
    f = open(f_name, 'wb')

    data = {}
    for i in range(len(DATABASE_SETS)):
        db = DATABASE_SETS[i]
        num_all = len(db)
        num_cur = 0
        for j in range(num_all):
            _filename = db[j]["query"]
            feature = save_fpfh_feature(_filename, dataset_prefix, save=False)
            data[_filename] = {}
            data[_filename]["fpfh"] = feature.data # np.ndarray

            num_cur += 1
            per = num_cur/num_all
            print(" -> dataset %d: "%i + "|%-*s|"%(20, ">" * int(20*per)) + " %.0f%%"%(per*100), end='\r')
        print()

    pickle.dump(data, f)
    f.close()

    print(" -> save as: " + f_name)
    return data


# data = { "filename" : {"fpfh" : feature_ndarray} }
def load_fpfh_pickle(dataset_prefix, database_file=""):
    f_name = get_feature_file(dataset_prefix)
    try:
        with open(f_name, 'rb') as f:
            data = pickle.load(f)
            return data
    except Exception as ex:
        data = save_fpfh_pickle(dataset_prefix, database_file)
        return data
    finally:
        print("Loading FPFH feature success.")


if __name__ == "__main__":
    DATASET_PREFIX = "university"
    DATABASE_FILE = config_reg.get_dataset_pickle_file(DATASET_PREFIX)
    data_fpfh = load_fpfh_pickle(DATASET_PREFIX, DATABASE_FILE)
