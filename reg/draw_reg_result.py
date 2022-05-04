import numpy as np
import os
import copy
import open3d
from config_reg import load_pts_file

# pts is a np.array, output the open3d's geometry
def convert_to_open3d_geometry(pts):
    s = open3d.geometry.PointCloud()
    s.points = open3d.utility.Vector3dVector(pts)
    return s


def paint_color(s, _type='s'):
    # light yellow [1, 0.706, 0]
    # sky blue [0, 0.651, 0.929]
    if _type == 's':
        s.paint_uniform_color([1.0, 0.4, 0.0])
    elif _type == 't':
        s.paint_uniform_color([0, 0.4, 1.0])


def draw_reg_from_file(_name, _s1_file, _s2_file, _transformation=None):
    _pts1 = load_pts_file(_s1_file)
    _pts2 = load_pts_file(_s2_file)
    _s1 = convert_to_open3d_geometry(_pts1)
    _s2 = convert_to_open3d_geometry(_pts2)
    paint_color(_s1, 's')
    paint_color(_s2, 't')
    if _transformation is not None:
        _s2.transform(_transformation)
    open3d.io.write_point_cloud(_name, _s1 + _s2)


def draw_ply_from_file(_name, _s_file, _type='t'):
    _pts = load_pts_file(_s_file)
    _s = convert_to_open3d_geometry(_pts)
    paint_color(_s, _type)
    open3d.io.write_point_cloud(_name, _s)


def draw_keypoints_from_file(_name, _s_file, ind):
    _pts = load_pts_file(_s_file)
    _s = convert_to_open3d_geometry(_pts)
    _s_k = _s.select_down_sample(ind)
    _s_nk = _s.select_down_sample(ind, invert=True)

    _s_k.paint_uniform_color([1, 0, 0])
    _s_nk.paint_uniform_color([0.8, 0.8, 0.8])
    open3d.io.write_point_cloud(_name, _s_k + _s_nk)
