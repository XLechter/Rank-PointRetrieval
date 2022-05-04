import open3d
import numpy as np
import os
import sys
import copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree
import time
import cv2

import get_pcd_dataset
import get_keypoints
import get_fpfh_feature
#import get_ppf_feature
from points_to_panorama import point_cloud_to_panorama, point_cloud_to_panorama_sphere
import config_reg
import tracemalloc

# current dataset's name for indexing feature file
DATASET_PREFIX = None
# type for evaluating registration results
type_reg = 'vc'
# if true, using PPF feature to do reg, or using FPFH
with_PPF_feature = False
# which type of keypoints to use, and how much
num_keypts = None
type_keypts = None
# if true, output logs when evaluating registration
LOG = False
# if true, output time using
LOG_TIME = False
# for debugging
DEBUG = False

# pre-load dataset
data_pcd = None
data_ppf = None
data_fpfh = None
data_kids = None

WITH_NOISES = True
SIGMA = 0.01


def init(dataset_prefix='', keypoints=32, keytype='rand', with_ppf_feature=True,
         reg_type = 'vc', with_noise=False,
      sigma=0.01, log_time=False):
    global DATASET_PREFIX
    DATASET_PREFIX = dataset_prefix

    global type_reg, num_keypts, type_keypts
    type_reg = reg_type
    num_keypts = keypoints
    type_keypts = keytype

    global data_pcd
    data_pcd = get_pcd_dataset.load_pcd_dataset(DATASET_PREFIX)

    global data_kids
    if num_keypts > 0:
        get_keypoints.init(keypoints=keypoints, keytype=keytype)
        data_kids = get_keypoints.load_keypoints_pickle(DATASET_PREFIX)

    global with_PPF_feature, data_ppf, data_fpfh
    with_PPF_feature = with_ppf_feature
    if with_ppf_feature:
        get_ppf_feature.init(keypoints=keypoints, keytype=keytype)
        data_ppf = get_ppf_feature.load_ppf_pickle(DATASET_PREFIX)
    else:
        data_fpfh = get_fpfh_feature.load_fpfh_pickle(DATASET_PREFIX)

    global WITH_NOISES
    WITH_NOISES = with_noise

    global SIGMA
    SIGMA = sigma

    global LOG_TIME
    LOG_TIME = log_time


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    # print('sigma', sigma)
    # if sigma > 0.01:
    #     clip = clip/0.01*0.05
    #     print('clip', clip)
    clip = 1.0
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

# ------------------------------------------------------------------
# pcd is a np.array, output the open3d's geometry

def convert_to_open3d_geometry(file):

    pc = data_pcd[file]["pcd"]
    if WITH_NOISES:
        pc_expand = np.expand_dims(pc, axis=0)
        pc = jitter_point_cloud(pc_expand, SIGMA).squeeze()
    num_pts = pc.shape[0]

    s = open3d.geometry.PointCloud()
    s.points = open3d.utility.Vector3dVector(pc)

    if with_PPF_feature:
        data = data_ppf[file]

        if num_keypts > 0 and num_keypts < num_pts:
            keyids = data_kids[file]["kids"]

        # s_down = s.select_down_sample(keyids) # ERROR FUNCTION with random index!
        s_down = open3d.geometry.PointCloud()
        s_down.points = open3d.utility.Vector3dVector(np.array(s.points)[keyids,:])

        s_feature = open3d.registration.Feature()
        s_feature.data = data["ppf"]

        return pc, keyids, s, s_down, s_feature

    else:
        # s_fpfh.data is a (33, n) nd array, n is the points number
        s_f_np = data_fpfh[file]["fpfh"]

        # using keypoints to do registration, so here select keypoints' feature
        if num_keypts > 0 and num_keypts < num_pts:
            keyids = data_kids[file]["kids"]

            # s_down = s.select_down_sample(keyids) # ERROR FUNCTION
            s_down = open3d.geometry.PointCloud()
            s_down.points = open3d.utility.Vector3dVector(np.array(s.points)[keyids,:])

            # print( np.asarray(s_down.points).shape )  # (keypts, 3)
            s_f_down_np = s_f_np[:, keyids]
            # print(s_f_down_np.shape)  # (33, keypts)
            s_f_down = open3d.registration.Feature()
            s_f_down.data = s_f_down_np

            return pc, keyids, s, s_down, s_f_down

        else:
            keyids = np.arange(0, num_pts)

            s_f = open3d.registration.Feature()
            s_f.data = s_f_np

            return pc, keyids, s, s, s_f


# ------------------------------------------------------------------
# using fast global registration
def run_fgr(s1, s2, s1_f, s2_f):
    distance_threshold = 0.05
    start = time.time()
    reg = open3d.registration.registration_fast_based_on_feature_matching(
        s2, s1, s2_f, s1_f,
        open3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    if LOG_TIME:
        print("FGR time %.3f sec" % (time.time() - start))
    # start = time.time()
    # add a ICP registration for a more accurate result
    # threshold = 0.05
    # reg = open3d.registration.registration_icp(s2, s1, threshold, reg.transformation,
    #    open3d.registration.TransformationEstimationPointToPoint(),
    #    open3d.registration.ICPConvergenceCriteria(max_iteration = 50))
    # print("ICP time %.3f sec" % (time.time() - start))
    # trans_init = np.asarray([[1., 0., 0., 0.],
    #                          [0., 1., 0., 0.],
    #                          [0., 0., 1., 0.],
    #                          [0., 0., 0., 1.]])
    return reg


# ------------------------------------------------------------------
def run_ransac(s1, s2, s1_f, s2_f):
    distance_threshold = 0.05
    reg = o3d.registration.registration_ransac_based_on_feature_matching(
        s2, s1, s2_f, s1_f, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(1000000, 100))
    return reg


# ------------------------------------------------------------------
def draw_registration_result(s1, s2, transformation, name):
    s1_temp = copy.deepcopy(s1)
    s2_temp = copy.deepcopy(s2)
    s1_temp.paint_uniform_color([1, 0.706, 0])
    s2_temp.paint_uniform_color([0, 0.651, 0.929])
    s2_temp.transform(transformation)
    open3d.io.write_point_cloud(name, s1_temp + s2_temp)


# ------------------------------------------------------------------
# pc1 : the origin pts of the source
# pc2_new : the target pts after reg.transformation
# result : the dict to store results (pointer)
def evaluate_with_rigid_match(pc1, pc2_new, result):
    tree1 = KDTree(pc1)
    dist2to1, ind2to1 = tree1.query(pc2_new)
    tree2 = KDTree(pc2_new)
    dist1to2, ind1to2 = tree2.query(pc1)

    # for ith point in pc1, the corresponding point in pc2 is ind1to2[i]
    # for every point, add a flag to set if it has a rigid registration
    reg1to2_score = np.ndarray((pc1.shape[0], 1), dtype=np.float32)
    for i in range(0, ind1to2.shape[0]):
        reg1to2_score[i] = 0

    # check if the point has a rigid corresponding match
    max_match_d = 0.05  # if the distance is bigger than this value, it's not a good match
    # fail_match_d = 0.30  # if the distance is too big, than it's a failure match
    for i in range(0, ind1to2.shape[0]):
        reg_i = ind1to2[i]
        # if dist1to2[i] > fail_match_d:
        #     reg1to2_score[i] = 0
        #     continue
        if ind2to1[reg_i] == i:
            if dist1to2[i] <= max_match_d:
                reg1to2_score[i] = 1.0
                continue
            # reg1to2_score[i] = (fail_match_d - dist1to2[i])
        # else:
            # reg1to2_score[i] = -0.02

    v_match = sum(reg1to2_score)[0]
    r_match = v_match / reg1to2_score.shape[0]

    result['or'] = r_match


# ------------------------------------------------------------------
# pc1 : the origin pts of the source
# pc2_new : the target pts after reg.transformation
# result : the dict to store results (pointer)
def evaluate_with_panorama(pc1, pc2_new, result):
    # v_res = 1.0 # smaller, the height bigger
    # h_res = 1.5 # smaller, the width bigger
    # v_fov = (-10.0, 35.0)
    # depth1 = point_cloud_to_panorama(pc1, v_res=v_res, h_res=h_res, v_fov=v_fov, y_fudge=0)
    # depth2 = point_cloud_to_panorama(pc2_new, v_res=v_res, h_res=h_res, v_fov=v_fov, y_fudge=0)

    cz = 0.0
    z_zoom = 1.0
    h = 90

    depth1 = point_cloud_to_panorama_sphere(pc1, cz=cz, z_zoom=z_zoom, h=h)
    depth2 = point_cloud_to_panorama_sphere(pc2_new, cz=cz, z_zoom=z_zoom, h=h)

    # depth1 = filter_depth_with_filling(depth1)
    # depth2 = filter_depth_with_filling(depth2)

    mask1 = depth1 > 0
    mask2 = depth2 > 0
    mask = mask1 & mask2  # mask for pixels with depth
    n = np.sum(mask1 | mask2)

    dis = depth2 - depth1
    dis = np.abs(dis)

    occ = dis > 0.12
    occ = occ & mask
    lap = dis <= 0.12
    lap = lap & mask

    result['vc_occ'] = np.sum(occ)
    result['vc_lap'] = np.sum(lap)

    result['vc'] = (result['vc_lap'] - result['vc_occ'] * 0.5) / n

    if LOG and DEBUG:
        f1 = os.path.basename(result['file1'])[0:-4]
        f2 = os.path.basename(result['file2'])[0:-4]
        DEBUG_PATH = './DEBUG'
        if not os.path.exists(DEBUG_PATH):
            os.makedirs(DEBUG_PATH)

        # np.savetxt(f"{DEBUG_PATH}/{f1}_{f2}_dis.txt", dis)

        img1 = np.zeros((depth1.shape[0], depth1.shape[1], 4))
        img1[mask1] = (0,200,200,255)
        #cv2.imwrite(f"{DEBUG_PATH}/{f1}_{f2}_f1.png", img1)

        img2 = np.zeros((depth2.shape[0], depth2.shape[1], 4))
        img2[mask2] = (200,200,0,255)
        #cv2.imwrite(f"{DEBUG_PATH}/{f1}_{f2}_f2.png", img2)

        img = np.zeros((depth1.shape[0], depth1.shape[1], 4))
        img += img1
        img += img2
        img[occ] = (255,0,0,255) # blue for points occluding
        img[lap] = (0,0,255,255) # red for points overlapping
        cv2.imwrite(f"{DEBUG_PATH}/{result['cur']}_{result['id2']}{'*' if result['gt'] else ''}.png", img)


def filter_depth_with_filling(depth_img):
    d_fill = 5
    depth_new = np.zeros((depth_img.shape[0], depth_img.shape[1]), dtype=np.float)
    depth_c = np.zeros((depth_img.shape[0], depth_img.shape[1]), dtype=np.float)
    # find pixel that has depth
    ys, xs = np.where(depth_img > 0)
    for i, y in enumerate(ys):
        x = xs[i]
        depth_cur = depth_img[y, x]

        # using the depth to do the filling in 3*3 area
        for y_d in range(d_fill):
            y_ = y + y_d - d_fill // 2
            if y_ < 0 or y_ >= depth_new.shape[0]:
                continue
            for x_d in range(d_fill):
                x_ = x + x_d - d_fill // 2
                if x_ < 0 or x_ >= depth_new.shape[1]:
                    continue
                depth_new[y_, x_] += depth_cur
                depth_c[y_, x_] += 1

    depth_c[depth_c == 0] = 1

    depth_new = depth_new * 1.0 / depth_c
    return depth_new


# ------------------------------------------------------------------
# files_data = [{'file': filename, 'gt':Bool}]
def evaluate_matchs(file1, files_data, log=True):
    global LOG
    LOG = log
    tracemalloc.start()
    results = []
    # pc = np.ndarray((n,3))
    pc1, s1_keyids, s1, s1_down, s1_f = convert_to_open3d_geometry(file1)

    index_max = len(files_data)
    for index, file2_data in enumerate(files_data):
        if log:
            print("processing: |%-*s|" % (index_max, ">" * (index+1)), end='\r')
        result = {}
        result['file1'] = file1
        result['file2'] = file2_data['file']
        result['id2'] = file2_data['id2']
        result['gt'] = file2_data['gt']
        result['cur'] = file2_data['cur']

        pc2, s2_keyids, s2, s2_down, s2_f = convert_to_open3d_geometry(file2_data['file'])

        # do registration
        reg = run_fgr(s1_down, s2_down, s1_f, s2_f)
        result['trans'] = reg.transformation

        # apply registration's result
        s2_new = copy.deepcopy(s2)
        s2_new.transform(reg.transformation)
        pc2_new = np.asarray(s2_new.points)

        # check the registration result
        # however, no results for fgr
        cor_set = np.asarray(reg.correspondence_set)
        result['fgr_setc'] = cor_set.shape[0]
        result['fgr_fitness'] = reg.fitness
        result['fgr_rmse'] = reg.inlier_rmse

        # using custom standard
        if type_reg == 'or':
            start = time.time()
            evaluate_with_rigid_match(pc1, pc2_new, result)
            if LOG_TIME:
                print("rigid match time %.3f sec" % (time.time() - start))
            result['reg_score'] = result['or']

        if type_reg == 'vc':
            start = time.time()
            evaluate_with_panorama(pc1, pc2_new, result)
            if LOG_TIME:
                print("panorama time %.3f sec" % (time.time() - start))
            result['reg_score'] = result['vc']

        results.append(result)
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()

    return results
