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

    n = num_cls
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
    palette = np.array(palette).reshape(3, -1).transpose()/255.
    return palette


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
                # bbox_file = pts_file.replace('.pts', '.bbox')
                # if os.path.exists(bbox_file):
                #     with open(bbox_file) as f:
                #         bbox = np.loadtxt(f).astype(np.int)
                # else:
                #     bbox = None
                bbox = None
                if bbox is not None:
                    palette = get_palette(bbox.shape[0])
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