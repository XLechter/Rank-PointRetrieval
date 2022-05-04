import numpy as np
import os
import sys

import config_reg

# ------------------------------------------------------------------
# data = { "filename" : {"pcd" : ndarray} }
def load_pcd_dataset(dataset_prefix, database_file=""):
    DATABASE_FILE = config_reg.get_dataset_pickle_file(dataset_prefix)
    if database_file != "":
        DATABASE_FILE = database_file

    try:
        DATABASE_SETS = config_reg.get_sets_dict(DATABASE_FILE)
    except Exception as ex:
        print("[ERROR] Load database failed! No such file: " + DATABASE_FILE)
        print(f"[ERROR] {type(ex)} | {ex.args!r}")
        return {}

    print("Loading pointclouds: " + DATABASE_FILE)

    data = {}
    for i in range(len(DATABASE_SETS)):
        db = DATABASE_SETS[i]
        num_all = len(db)
        num_cur = 0
        for j in range(num_all):
            _filename = db[j]["query"]
            pcd = config_reg.load_pts_file(_filename)
            data[_filename] = {}
            data[_filename]["pcd"] = pcd  # np.ndarray

            num_cur += 1
            per = num_cur/num_all
            print(" -> dataset %d: "%i + "|%-*s|"%(20, ">" * int(20*per)) + " %.0f%%"%(per*100), end='\r')
        print()

    print("Loading pointclouds success.")
    return data
