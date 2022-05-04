import sys
import numpy as np
import heapq
import time
import pickle
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import open3d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree

import config_reg
import draw_reg_result
PATH_SAVE_ROOT = config_reg.DATA_FOLDER + '/DATA_KEYPOINTS'
if not os.path.exists(PATH_SAVE_ROOT):
    os.makedirs(PATH_SAVE_ROOT)

# global
DEBUG = False
num_keypoints = None
type_keypts = None


def init(keypoints=128, keytype='rand'):
    global num_keypoints, type_keypts
    num_keypoints = keypoints
    type_keypts = keytype


# ------------------------------------------------------------------
def filter_solo_point_in_pts(pts, debug_file=None):
    tree1 = KDTree(pts)
    k = 6
    dist, ind = tree1.query(pts, k=k)
    if DEBUG and debug_file is not None:
        np.savetxt(debug_file+'_dist.txt', dist, fmt='%f',delimiter=',')

    delete_ids = []
    for i in range(pts.shape[0]):
        if dist[i][k-1] > 0.04:
            delete_ids.append(i)

    pts = np.delete(pts, delete_ids, axis=0)
    return pts


# ------------------------------------------------------------------
def load_keypoints_fps(filenames, num_):
    keypts_inds = []
    for filename in filenames:
        pts = config_reg.load_pts_file(filename)
        keyids = farthest_point_sample(pts, num_)
        keypts_inds.append(keyids)

    return keypts_inds


# pts - point cloud,  np.ndarray( (N,3) )
# n - number of samples
# return [index] of the points on pts
def farthest_point_sample(pts, n_sample):
    n_total, _ = pts.shape
    # save the (n_sample)th farthest point's index
    centroids = np.zeros((n_sample,), dtype=np.long)
    # save all points' max distance
    distance = np.ones((n_total,)) * 1e10
    # choose a random one as init point
    # farthest = np.random.randint(0, n_total, (1,))[0]
    # choose the first point as init point
    farthest = 0
    # get the 1st to (n_sample)th points with biggest distance
    for i in range(n_sample):
        centroids[i] = farthest
        centroid = pts[farthest, :]  # current farthest point (x,y,z)
        dist = np.sum((pts - centroid) ** 2, -1)  # distance between current point and others
        mask = dist < distance
        distance[mask] = dist[mask]  # save the max distance for all points
        # get the next point with max distance
        dist_max = np.max(distance)
        a = np.where(distance == dist_max)
        farthest = a[0][0]
    return centroids


# ------------------------------------------------------------------
def load_keypoints_vds(filenames, num_):
    keyidss_rand = load_keypoints_rand(filenames, num_)

    keypts_inds = []
    for i, filename in enumerate(filenames):
        pts = config_reg.load_pts_file(filename)
        pts = filter_solo_point_in_pts(pts)
        keyids = voxel_downsample(pts, voxel_size = 0.1)

        kids = np.append(keyids, keyidss_rand[i])
        kids = np.unique(kids)

        if DEBUG:
            s = os.path.basename(filenames[i])[0:-4]
            _path = os.path.join(PATH_SAVE_ROOT, f'DEBUG-vds{num_}')
            if not os.path.exists(_path):
                os.makedirs(_path)
            draw_reg_result.draw_keypoints_from_file(_path+'/'+s+'_vds.ply', filenames[i], keyids)
            draw_reg_result.draw_keypoints_from_file(_path+'/'+s+'_all.ply', filenames[i], kids)
            # np.savetxt(_path+'/'+s+'_keys.txt', keyids)

        keypts_inds.append(kids[0:num_])

    return keypts_inds


def voxel_downsample(pts, voxel_size = 0.1):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size)

    voxel_down_points = np.array(pcd.points)

    indexes = []
    for i in range(len(voxel_down_points)):
        residuals = pts - voxel_down_points[i]

        residuals = np.sum(np.absolute(residuals), axis=1)
        index = residuals.argmin()
        indexes.append(index)

    indexes = np.unique(indexes)
    return indexes


# ------------------------------------------------------------------
def load_keypoints_rand(filenames, num_):
    keypts_inds = []
    for filename in filenames:
        pts = config_reg.load_pts_file(filename)
        num_pts = pts.shape[0]
        keyids = np.random.choice(range(num_pts), num_, replace=False)
        keypts_inds.append(keyids)

    return keypts_inds


# ------------------------------------------------------------------
# get the pointcloud's keypoints by keytype
# ------------------------------------------------------------------
def load_keypoints(filenames):
    global type_keypts
    try:
        return globals().get(f'load_keypoints_{type_keypts}')(filenames, num_keypoints)
    except Exception as ex:
        print(f"[ERROR] Load keypoints failed! Maybe no function load_keypoints_{type_keypts}, or something wrong in the function.")
        print(f"[ERROR] {type(ex)} | {ex.args!r}")
        print(f"Using load_keypoints_rand for the alternative.")
        type_keypts = 'rand'
        return load_keypoints_rand(filenames, num_keypoints)


# ------------------------------------------------------------------
# get a dataset's keypoints
# ------------------------------------------------------------------
def get_keypoints_file(dataset_prefix):
    return PATH_SAVE_ROOT + "/" + dataset_prefix + "_%s%d.pickle" % (type_keypts, num_keypoints)


# save the dataset's keypoints in one file
def save_keypoints_pickle(dataset_prefix, database_file=""):
    DATABASE_FILE = config_reg.get_dataset_pickle_file(dataset_prefix)
    if database_file != "":
        DATABASE_FILE = database_file

    try:
        DATABASE_SETS = config_reg.get_sets_dict(DATABASE_FILE)
    except Exception as ex:
        print("[ERROR] Load database failed! No such file: " + DATABASE_FILE)
        print(f"[ERROR] {type(ex)} | {ex.args!r}")
        return {}

    print(f"Processing Keypoints({type_keypts}-{num_keypoints}): " + DATABASE_FILE)

    f_name = get_keypoints_file(dataset_prefix)
    f = open(f_name, 'wb')

    data = {}
    batch = 12  # to fit PCAN, batch numbers pointclouds will be calculated in one time
    for i in range(len(DATABASE_SETS)):
        db = DATABASE_SETS[i]
        num_all = len(db)
        num_cur = 0
        for j in range((num_all+batch-1) // batch):
            _filenames = []
            for k in range(j * batch, (j + 1) * batch):
                if k >= len(db):
                    break
                _filename = db[k]["query"]
                _filenames.append(_filename)
            kidss = load_keypoints(_filenames)

            for _i, _file in enumerate(_filenames):
                data[_file] = {}
                data[_file]["kids"] = kidss[_i]  # np.ndarray

            num_cur += len(_filenames)
            per = num_cur/num_all
            print(" -> dataset %d: "%i + "|%-*s|"%(20, ">" * int(20*per)) + " %.0f%%"%(per*100), end='\r')
        print()

    pickle.dump(data, f)
    f.close()

    print(" -> save as: " + f_name)
    return data


# data = { "filename" : {"kids" : ndarray((n,))} }
def load_keypoints_pickle(dataset_prefix, database_file=""):
    f_name = get_keypoints_file(dataset_prefix)
    try:
        with open(f_name, 'rb') as f:
            data = pickle.load(f)
            return data
    except Exception as ex:
        data = save_keypoints_pickle(dataset_prefix, database_file)
        return data
    finally:
        print(f"Loading Keypoints({type_keypts}-{num_keypoints}) success.")


# ------------------------------------------------------------------
if __name__ == "__main__":
    DEBUG = True
    init(keypoints=20, keytype='512')
    DATASET_PREFIX = "university"
    DATABASE_FILE = config_reg.get_dataset_pickle_file(DATASET_PREFIX)
    data = load_keypoints_pickle(DATASET_PREFIX, DATABASE_FILE)
    print(data)
