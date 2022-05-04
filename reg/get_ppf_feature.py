import numpy as np
import open3d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree
import torch
import torch.nn as nn
import importlib
import sys
import os
import pickle

import config_reg
import get_keypoints

#PATH_SAVE_ROOT = config_reg.DATA_FOLDER + '/DATA_FEATURES_PPF'
# for lacking of storege, change the path to store ppf featrue
#  (which is significantly larger than FPFH)
PATH_SAVE_ROOT = '/mnt/data/zwx/PPF-FoldNet' + '/DATA_FEATURES_PPF'
if not os.path.exists(PATH_SAVE_ROOT):
    os.makedirs(PATH_SAVE_ROOT)

try:
    ppf_root_path = "/home/user/zwx/PPF-FoldNet"
    sys.path.append(ppf_root_path)
    from input_preparation import collect_local_neighbor, build_local_patch_with_pf, build_local_patch

    # pnv_root_path = "/home/user/zwx/PointNetVlad-Pytorch"
    # sys.path.append(pnv_root_path)
    # import models.PointNetVlad_point_feature as PNV
except Exception as ex:
    print(f"[ERROR] {type(ex)} | {ex.args!r}")

num_keypts = 32  # the number of patchs (also the keypoints' number)
num_points_per_patch = 16 ** 2  # number of points in a patch
num_features = 512 * 1  # the dimension of the feature

type_keypts = None
data_kids = None

ppf_model = None
pnv_model = None
pnv_device = None


def init(keypoints=32, keytype='rand'):
    global num_keypts, type_keypts
    num_keypts = keypoints
    type_keypts = keytype

    # ------------------------------------------------------------------
    # get PPF-FoldNet feature
    # ------------------------------------------------------------------
    ppf_path = ppf_root_path + '/snapshot/PPF-FoldNet_1120'

    print('ppf_path', ppf_path)
    global ppf_model
    try:
        ppf_module_name = 'models'
        ppf_module_file = ppf_path + '/model.py'
        ppf_module_spec = importlib.util.spec_from_file_location(ppf_module_name, ppf_module_file)
        ppf_module = importlib.util.module_from_spec(ppf_module_spec)
        ppf_module_spec.loader.exec_module(ppf_module)

        ppf_model = ppf_module.PPFFoldNet(num_keypts, num_points_per_patch)
        ppf_model.load_state_dict(torch.load(ppf_path + '/models/oxford_best.pkl'))
        ppf_model.eval()
        print("Init PPF success from " + ppf_module_file)

    except Exception as ex:
        print("[ERROR] Load PPF-FoldNet failed!")
        print(f"[ERROR] {type(ex)} | {ex.args!r}")

    # ------------------------------------------------------------------
    # get PointNetVlad-Pytorch feature
    # ------------------------------------------------------------------
    # pnv_path = pnv_root_path + "/trained_models"
    # pnv_model_file = "vlad.pth"
    #
    # global pnv_model, pnv_device
    # try:
    #     pnv_model = PNV.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
    #                                  output_dim=256, num_points=4096)
    #
    #     pnv_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     pnv_model = pnv_model.to(pnv_device)
    #
    #     resume_filename = pnv_path + '/' + pnv_model_file
    #     checkpoint = torch.load(resume_filename)
    #     saved_state_dict = checkpoint['state_dict']
    #     pnv_model.load_state_dict(saved_state_dict)
    #
    #     pnv_model = nn.DataParallel(pnv_model)
    #     pnv_model.eval()
    #     print("Init PointNetVlad success from " + resume_filename)
    #
    # except Exception as ex:
    #     print("[ERROR] Load PointNetVlad-Pytorch failed!")
    #     print(f"[ERROR] {type(ex)} | {ex.args!r}")


# ------------------------------------------------------------------
# get a pointcloud's ppf feature
# ------------------------------------------------------------------
# pcd : open3d.geometry
# pts : np.ndarray((n, 3))
def get_ppf_feature(pcd, keyids):
    pts = np.asarray(pcd.points)

    #_keypts = np.asarray(pcd.select_down_sample(keyids).points) # ERROR FUNCTION
    _keypts = np.array(pcd.points)[keyids,:]

    _kdtree = open3d.geometry.KDTreeFlann(pcd)
    _keypts_id = []
    for i in range(_keypts.shape[0]):
        _, id, _ = _kdtree.search_knn_vector_3d(_keypts[i], 1)
        _keypts_id.append(id[0])
    # ref_pcd = pcd.select_down_sample(_keypts_id) # ERROR FUNCTION
    ref_pcd = open3d.geometry.PointCloud()
    ref_pcd.points = open3d.utility.Vector3dVector(np.array(pcd.points)[_keypts_id,:])

    # get patchs
    local_patches = build_ppf_input(ref_pcd, pcd, num_points_per_patch)  # [num_keypts, 1024, 4]
    local_patches_np = local_patches.astype(np.float32)

    # calculate features
    input_ = torch.Tensor(local_patches_np).cuda()
    model = ppf_model.cuda()
    desc = model.encoder(input_)
    desc_ = desc.detach().cpu().numpy().squeeze()

    s_features = open3d.registration.Feature()
    s_features.data = desc_.transpose()

    # return keypoints' indexes in origin pointcloud, feature from PPF
    return s_features


def build_ppf_input(ref_pcd, pcd, num_points_per_patch):
    neighbor = collect_local_neighbor(ref_pcd, pcd, vicinity=0.3, num_points_per_patch=num_points_per_patch)

    # raw PPF
    local_patches = build_local_patch(ref_pcd, pcd, neighbor)

    # with PointNetVlad feature
    #pf = get_pnv_feature(pcd)
    #local_patches = build_local_patch_with_pf(ref_pcd, pcd, neighbor, pf)

    return local_patches


# def get_pnv_feature(pcd):
#     queries = np.asarray(pcd.points)
#     queries = np.expand_dims(queries, 0)
#     with torch.no_grad():
#         feed_tensor = torch.from_numpy(queries).float()
#         feed_tensor = feed_tensor.unsqueeze(1)
#         feed_tensor = feed_tensor.to(pnv_device)
#         _, out = pnv_model(feed_tensor)
#
#     out = out.detach().cpu().numpy()
#     out = np.squeeze(out)
#     out = np.transpose(out, (1, 0))
#
#     return out


# filename : pointcloud file in the DATASET_ROOT path
def save_ppf_feature(filename, dataset_prefix, save=True):
    pts = config_reg.load_pts_file(filename)
    s = open3d.geometry.PointCloud()
    s.points = open3d.utility.Vector3dVector(pts)

    keyids = data_kids[filename]["kids"]
    s_feature = get_ppf_feature(s, keyids)
    # print( type(s_feature.data) ) # np.ndarray

    if save:
        dir_ = os.path.join(PATH_SAVE_ROOT, dataset_prefix, os.path.dirname(filename))
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        file_save = os.path.join(PATH_SAVE_ROOT, dataset_prefix, filename + ".npz")
        np.savez(file_save, keyids=inds, ppf_f=s_feature.data)
        # s_feature.data.tofile(file_save)

    return s_feature


def load_ppf_feature(filename, dataset_prefix):
    file_f = os.path.join(PATH_SAVE_ROOT, dataset_prefix, filename + ".npz")
    if os.path.isfile(file_f):
        # fs = np.fromfile(file_f, dtype=np.float)
        r = np.load(file_f)

        s_feature = open3d.registration.Feature()
        s_feature.data = r['ppf_f']

        return s_feature
    else:
        s_feature = save_ppf_feature(filename, dataset_prefix)
        return s_feature


# ------------------------------------------------------------------
# get a dataset's ppf
# ------------------------------------------------------------------
def get_feature_file(dataset_prefix):
    surfix = "_%s%s_%s_%s" % (type_keypts, num_keypts, num_points_per_patch, num_features)
    return PATH_SAVE_ROOT + "/" + dataset_prefix + surfix + ".pickle"


# save the dataset's ppf features in one file
def save_ppf_pickle(dataset_prefix, database_file=""):
    DATABASE_FILE = config_reg.get_dataset_pickle_file(dataset_prefix)
    if database_file != "":
        DATABASE_FILE = database_file

    try:
        DATABASE_SETS = config_reg.get_sets_dict(DATABASE_FILE)
    except Exception as ex:
        print("[ERROR] Load database failed! No such file: " + DATABASE_FILE)
        print(f"[ERROR] {type(ex)} | {ex.args!r}")
        return {}

    print(f"Processing PPF features({type_keypts}{num_keypts}-{num_points_per_patch}-{num_features}): " + DATABASE_FILE)

    f_name = get_feature_file(dataset_prefix)
    f = open(f_name, 'wb')

    data = {}
    for i in range(len(DATABASE_SETS)):
        db = DATABASE_SETS[i]
        num_all = len(db)
        num_cur = 0
        for j in range(num_all):
            _filename = db[j]["query"]
            feature = save_ppf_feature(_filename, dataset_prefix, save=False)
            data[_filename] = {}
            data[_filename]["ppf"] = feature.data  # np.ndarray

            num_cur += 1
            per = num_cur / num_all
            print(" -> dataset %d: " % i + "|%-*s|" % (20, ">" * int(20 * per)) + " %.0f%%" % (per * 100), end='\r')
        print()

    pickle.dump(data, f)
    f.close()

    print(" -> save as: " + f_name)
    return data


# data = { "filename" : {"ppf" : feature_ndarray} }
def load_ppf_pickle(dataset_prefix, database_file=""):
    # ------------------------------------------------------------------
    # get keypoints
    # ------------------------------------------------------------------
    global data_kids
    get_keypoints.init(keypoints=num_keypts, keytype=type_keypts)
    data_kids = get_keypoints.load_keypoints_pickle(dataset_prefix)
    # ------------------------------------------------------------------
    # load ppf feature
    # ------------------------------------------------------------------
    f_name = get_feature_file(dataset_prefix)
    try:
        with open(f_name, 'rb') as f:
            data = pickle.load(f)
            return data
    except Exception as ex:
        data = save_ppf_pickle(dataset_prefix, database_file)
        return data
    finally:
        print(f"Loading PPF feature({type_keypts}{num_keypts}-{num_points_per_patch}-{num_features}) success.")


if __name__ == "__main__":
    init(keypoints=256, keytype='rand')

    DATASET_PREFIX = "university"
    DATABASE_FILE = config_reg.get_dataset_pickle_file(DATASET_PREFIX)
    load_ppf_pickle(DATASET_PREFIX, DATABASE_FILE)
