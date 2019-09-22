import time
import argparse
import os
import sys
import pptk
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../utils'))

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='log', help='Log dir [default: log]')
parser.add_argument('--eval_dir', type=str, default='eval', help='Eval dir [default: eval]')
parser.add_argument('--visu_dir', type=str, default=None, help='Visu dir [default: None, meaning no visu]')
FLAGS = parser.parse_args()

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    print('ERROR: log_dir %s does not exist! Please Check!' % LOG_DIR)
    exit(1)
LOG_DIR = os.path.join(LOG_DIR, FLAGS.eval_dir)
if FLAGS.visu_dir is not None:
    VISU_DIR = os.path.join(LOG_DIR, FLAGS.visu_dir)

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls+1
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    palette = np.array(palette).reshape(3, -1).transpose()[1:]/255.
    new_palette = np.array(
        [[109, 75, 186],
         [102, 206, 82],
         [138, 60, 186],
         [110, 180, 51],
         [189, 88, 219],
         [45, 153, 46],
         [215, 116, 237],
         [70, 199, 106],
         [203, 49, 160],
         [174, 195, 52],
         [67, 107, 234],
         [200, 177, 50],
         [58, 88, 196],
         [145, 170, 57],
         [154, 105, 227],
         [80, 135, 40],
         [169, 72, 181],
         [56, 204, 148],
         [228, 53, 136],
         [75, 164, 86],
         [235, 99, 197],
         [37, 124, 63],
         [229, 45, 93],
         [78, 209, 202],
         [194, 64, 26],
         [111, 130, 236],
         [228, 124, 33],
         [85, 75, 170],
         [227, 162, 57],
         [94, 100, 193],
         [134, 193, 112],
         [187, 82, 175],
         [52, 159, 114],
         [161, 46, 125],
         [115, 194, 149],
         [220, 80, 142],
         [76, 103, 21],
         [195, 127, 229],
         [115, 133, 42],
         [109, 69, 153],
         [188, 174, 86],
         [59, 102, 177],
         [167, 128, 33],
         [127, 148, 229],
         [235, 108, 69],
         [83, 154, 221],
         [204, 60, 61],
         [45, 171, 171],
         [206, 66, 90],
         [84, 182, 226],
         [153, 84, 20],
         [48, 119, 171],
         [191, 112, 47],
         [91, 86, 156],
         [135, 131, 44],
         [226, 141, 226],
         [54, 108, 56],
         [229, 138, 205],
         [65, 90, 31],
         [138, 97, 179],
         [168, 183, 113],
         [125, 70, 141],
         [99, 145, 86],
         [194, 92, 151],
         [26, 100, 71],
         [216, 96, 128],
         [61, 136, 111],
         [168, 46, 84],
         [89, 114, 54],
         [173, 135, 214],
         [97, 97, 23],
         [204, 166, 231],
         [114, 95, 26],
         [179, 106, 176],
         [88, 92, 38],
         [238, 135, 176],
         [126, 129, 73],
         [141, 74, 132],
         [216, 182, 121],
         [80, 91, 143],
         [232, 151, 89],
         [121, 115, 179],
         [166, 72, 40],
         [155, 154, 214],
         [127, 89, 36],
         [205, 137, 181],
         [153, 123, 65],
         [157, 100, 145],
         [194, 150, 95],
         [154, 69, 106],
         [233, 157, 122],
         [134, 72, 99],
         [180, 117, 72],
         [239, 155, 158],
         [143, 77, 46],
         [237, 121, 113],
         [147, 76, 75],
         [202, 116, 125],
         [167, 72, 81],
         [197, 111, 94]]
    )/255.
    return new_palette[1:n]


def plot_box(coords, scales):
    scales = scales/2.
    #coords -> (z,x,y)
    num_point = 200
    box_pc = []
    p_basic = np.repeat(np.linspace(1, num_point, num_point)/num_point, 3).reshape(1, -1, 3)
    num_box = coords.shape[0]
    s0 = np.zeros([num_box, 3])
    s1 = np.zeros([num_box, 3])
    s2 = np.zeros([num_box, 3])
    s0[:,0] = scales[:,2]
    s1[:,1] = scales[:,0]
    s2[:,2] = scales[:,1]
    #p00 = coords - scales
    #p11 = coords + scales
    p0 = np.zeros([num_box, 3])
    p1 = np.zeros([num_box, 3])
    p0[:,0] = coords[:,2] - scales[:,2]
    p0[:,1] = coords[:,0] - scales[:,0]
    p0[:,2] = coords[:,1] - scales[:,1]
    p1[:,0] = coords[:,2] + scales[:,2]
    p1[:,1] = coords[:,0] + scales[:,0]
    p1[:,2] = coords[:,1] + scales[:,1]
    p2 = p0 + 2*s2
    p3 = p1 - 2*s2
    p4 = p0 + 2*s0
    p5 = p0 + 2*s1
    s0=s0.reshape(-1,1,3)
    s1=s1.reshape(-1,1,3)
    s2=s2.reshape(-1,1,3)
    p0=p0.reshape(-1,1,3)
    p1=p1.reshape(-1,1,3)
    p2=p2.reshape(-1,1,3)
    p3=p3.reshape(-1,1,3)
    p4=p4.reshape(-1,1,3)
    p5=p5.reshape(-1,1,3)
    box_pc.append(p0+2*s0*p_basic)
    box_pc.append(p0+2*s1*p_basic)
    box_pc.append(p0+2*s2*p_basic)
    box_pc.append(p1-2*s0*p_basic)
    box_pc.append(p1-2*s1*p_basic)
    box_pc.append(p1-2*s2*p_basic)
    box_pc.append(p2+2*s0*p_basic)
    box_pc.append(p2+2*s1*p_basic)
    box_pc.append(p3-2*s0*p_basic)
    box_pc.append(p3-2*s1*p_basic)
    box_pc.append(p4+2*s2*p_basic)
    box_pc.append(p5+2*s2*p_basic)
    box_pc = np.array(box_pc).swapaxes(0,1).reshape(-1,3)
    final = np.zeros(box_pc.shape)
    final[:,0] = box_pc[:,1]
    final[:,1] = box_pc[:,2]
    final[:,2] = box_pc[:,0]
    #final[:,0] = box_pc[:,2]
    #final[:,1] = box_pc[:,0]
    #final[:,2] = box_pc[:,1]
    #return np.array(box_pc).reshape(-1,3)#.swapaxes(1,0)
    return final


def convert(visu_dir):
    for root, dirs, files in os.walk(visu_dir):
        for file in files:
            if file.endswith('.pts'):
                pts_file = os.path.join(root, file)
                label_file = pts_file.replace('.pts', '.label')
                out_file = pts_file.replace('.pts', '.png')
                print('rendering: {}'.format(pts_file))
                with open(pts_file) as f:
                    pts = np.loadtxt(f)
                if os.path.exists(label_file):
                    print('rendering: {}'.format(label_file))
                    with open(label_file) as f:
                        label = np.loadtxt(f, dtype=np.bool)
                else:
                    label = None
                if label is not None:
                    pts = pts[label]
                pts = np.stack([pts[:, 2], pts[:, 0], pts[:, 1]], axis=1)
                v = pptk.viewer(pts)
                v.set(point_size=0.01, r=5, show_grid=False, show_axis=False, lookat=[.8, .8, .8])
                v.capture(out_file)
                time.sleep(0.5)
                v.close()
                print('saving: {}'.format(out_file))
                # print('camera LA:', v.get('lookat'))
                mask_file = pts_file.replace('.pts', '.mask')
                if os.path.exists(mask_file):
                    with open(mask_file) as f:
                        mask = np.loadtxt(f).astype(np.int)
                else:
                    mask = None
                if mask is not None:
                    palette = get_palette(len(np.unique(mask)))
                    mask_new = mask.copy()
                    for idx, ins_id in enumerate(np.unique(mask)):
                        mask_new[mask == ins_id] = idx
                    v = pptk.viewer(pts, palette[mask_new, :])
                    v.set(point_size=0.01, r=5, show_grid=False, show_axis=False, lookat=[.8, .8, .8])
                    mask_out_file = out_file.replace('pts', 'mask')
                    if not os.path.exists(os.path.dirname(mask_out_file)):
                        os.mkdir(os.path.dirname(mask_out_file))
                    v.capture(mask_out_file)
                    time.sleep(0.5)
                    print('saving: {}'.format(mask_out_file))
                    v.close()
                bbox_file = pts_file.replace('.pts', '.bbox')
                if os.path.exists(bbox_file):
                    with open(bbox_file) as f:
                        bbox = np.loadtxt(f).astype(np.float)
                else:
                    bbox = None
                if bbox is not None:
                    palette = get_palette(bbox.shape[0])
                    if len(bbox.shape) == 1:
                        bbox = bbox[None, :]
                    bbox_new = bbox.copy()
                    bbox_new = plot_box(bbox_new[:, :3], bbox_new[:, 3:])
                    bbox_new = bbox_new.reshape((bbox.shape[0], -1, 3))
                    bbox_new_ins_id = np.repeat(np.arange(bbox_new.shape[0])[:, None], bbox_new.shape[1], axis=-1)
                    bbox_new_ins_id = bbox_new_ins_id.reshape((-1))
                    bbox_new = bbox_new.reshape((-1, 3))
                    bbox_new_pts = np.stack([bbox_new[:, 2], bbox_new[:, 0], bbox_new[:, 1]], axis=1)
                    v = pptk.viewer(bbox_new_pts, palette[bbox_new_ins_id, :])
                    v.set(point_size=0.01, r=5, show_grid=False, show_axis=False, lookat=[.8, .8, .8])
                    bbox_out_file = out_file.replace('pts', 'bbox')
                    if not os.path.exists(os.path.dirname(bbox_out_file)):
                        os.mkdir(os.path.dirname(bbox_out_file))
                    v.capture(bbox_out_file)
                    time.sleep(0.5)
                    print('saving: {}'.format(bbox_out_file))
                    v.close()

convert(VISU_DIR)