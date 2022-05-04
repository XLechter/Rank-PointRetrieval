import os
import pickle
import numpy as np
# ------------------------------------------------------------------
# the root path of all datasets
#   all pointcloud data is put in this path,
#   and its relative path is stored in like "generating_queries/oxford_evaluation_database.pickle"
#    ["query"] = "oxford/2014-11-18-13-20-12/pointcloud_20m/1416317061558654.bin"
DATASET_ROOT = '/mnt/data1/zwx/benchmark_datasets'

# ------------------------------------------------------------------
# where to store datas for speed-up
DATA_FOLDER = '/mnt/data2/zwxx/PointNetVlad-Pytorch/DATAS'
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# ------------------------------------------------------------------
# get the dataset's pickle file
def get_dataset_pickle_file(dataset_prefix, type='database'):
    t = ''
    if type == 'database':
        t = '../generating_queries/%s_evaluation_database.pickle' % dataset_prefix
    elif type == 'query':
        t = '../generating_queries/%s_evaluation_query.pickle' % dataset_prefix
    return t

# ------------------------------------------------------------------
# read the pickle file
def get_queries_dict(pickle_file):
    # key:{'query':file,'positives':[files],'negatives':[files], 'neighbors':[keys]}
    with open(pickle_file, 'rb') as handle:
        queries = pickle.load(handle)
        print("Queries Loaded.")
        return queries

def get_sets_dict(pickle_file):
    #[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
    with open(pickle_file, 'rb') as handle:
        trajectories = pickle.load(handle)
        print("Trajectories Loaded.")
        return trajectories

# ------------------------------------------------------------------
def load_pts_file(filename):
    # returns Nx3 matrix
    pts = np.fromfile(os.path.join(DATASET_ROOT, filename), dtype=np.float64)
    if pts.shape[0] != 4096 * 3:
        print("Error in point cloud shape")
        return np.array([])

    pts = np.reshape(pts, (pts.shape[0] // 3, 3))
    return pts

def load_pts_files(filenames):
    ptss = []
    for filename in filenames:
        pts = load_pts_file(filename)
        if(pts.shape[0] != 4096):
            continue
        ptss.append(pts)
    ptss = np.array(ptss)
    return ptss

# ------------------------------------------------------------------
def add_params_for_parser(parser):
    parser.add_argument('--dataset_prefix', default='business',
                        help='Dataset\'s prefix [default: business]')

    parser.add_argument('--result_path', default='../results',
                        help='Where to store the result txt and logs. [default: ./results_eagle] ')
    parser.add_argument('--log', action="store_true",
                        help='Output logs for each pointcloud.')
    parser.add_argument('--log_ply', action="store_true",
                        help='If output log, output result plys for each pointcloud.')
    parser.add_argument('--surfix', default='',
                        help='Log\'s name surfix [default:  ]')

    parser.add_argument('--with_reg', action="store_true",
                        help='Using reg to do re-ranking')
    parser.add_argument('--reg', default='vc',
                        help='Type using for evaluate reg result [default: vc]' + \
                        '  or: Overlap Ratio; ' + \
                        '  vc: Visual Consistency')

    parser.add_argument('--with_ppf_feature', action="store_true",
                        help='Using ppf to do reg')

    parser.add_argument('--keypoints', type=int, default=0,
                        help='Keypoint number selected as keypoints [default: 0]' + \
                        'If 0, all points will be used to do registration.')
    parser.add_argument('--keytype', default='rand',
                        help='Type using for selecting keypoints [default: rand]' + \
           '  rand: random selection; ' + \
           '  fps: farthest point sampling; ' + \
           '  vds: voxel downsample '
    )

    parser.add_argument('--lamda1', type=float, default=1.0,
                        help='Weight for the model\'s feature [default: 1.0]')
    parser.add_argument('--lamda2', type=float, default=0.8,
                        help='Weight for registration [default: 0.8]')
    parser.add_argument('--with_noise', action="store_true",
                        help='Using reg to do re-ranking')
    parser.add_argument('--sigma', type=float, default=0.01,
                        help='Weight for sigma [default: 1.0]')


def get_result_txt_name(FLAGS):
    txt_flag = ''

    if FLAGS.with_reg:
        txt_flag += f'_{FLAGS.reg}'

        if FLAGS.with_ppf_feature:
            txt_flag += '_ppf'
        else:
            txt_flag += '_fpfh'

        if FLAGS.keypoints > 0:
            txt_flag += f'_{FLAGS.keytype}{FLAGS.keypoints}'
        else:
            txt_flag += f'_{FLAGS.keypoints}'

        txt_flag += f'_{FLAGS.lamda1}_{FLAGS.lamda2}'
        
        if FLAGS.with_noise:
            txt_flag += f'_sigma{FLAGS.sigma}'

    else:
        txt_flag += '_no_opt'

    t = '%s%s%s.txt' % (FLAGS.dataset_prefix, txt_flag, FLAGS.surfix)

    if not os.path.exists(FLAGS.result_path):
        os.makedirs(FLAGS.result_path)

    return os.path.join(FLAGS.result_path, t)
