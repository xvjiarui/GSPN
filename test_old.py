import tensorflow as tf
import numpy as np
import argparse
import importlib
import time
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import data_prep_old as  data_prep
import config_old as config
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import random
import copy
import glob
import json
import h5py
random.seed(0)

CONFIG = config.Config()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=CONFIG.NUM_POINT, help='Point Number in a Scene [default: 2048]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point_ins', type=int, default=CONFIG.NUM_POINT_INS, help='Point Number of an Instance [default: 512]')
parser.add_argument('--num_category', type=int, default=CONFIG.NUM_CATEGORY, help='Maximum Number of Categories [default: 3]')
parser.add_argument('--num_sample', type=int, default=CONFIG.NUM_SAMPLE, help='Number of Sampled Seed Points [default: 128]')
parser.add_argument('--model', default='model_mrcnn3d', help='Model name [default: model]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--category', type=str, help='category')
parser.add_argument('--level_id', type=int, help='level_id')
FLAGS = parser.parse_args()

CONFIG.NUM_POINT = FLAGS.num_point
CONFIG.NUM_POINT_INS = FLAGS.num_point_ins
CONFIG.NUM_CATEGORY = FLAGS.num_category
CONFIG.NUM_SAMPLE = FLAGS.num_sample
MODEL_PATH = FLAGS.model_path
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module

CAT = FLAGS.category
LEVEL = FLAGS.level_id
if not os.path.exists('./data/partnet/ins_seg_h5_for_detection/cache/{}-{}/'.format(CAT, LEVEL)):
    os.makedirs('./data/partnet/ins_seg_h5_for_detection/cache/{}-{}/'.format(CAT, LEVEL))
TRAIN_DATASET = data_prep.PartNetDataset('./data/partnet/ins_seg_h5_for_detection/{}-{}/'.format(CAT, LEVEL),
                                         './data/partnet/ins_seg_h5_for_detection/cache/{}-{}/train_{}_00_cache.npz'.format(CAT, LEVEL, CONFIG.NUM_POINT),
                                         data_type='train', npoint=CONFIG.NUM_POINT, npoint_ins=CONFIG.NUM_POINT_INS, is_augment=True, permute_points=True,
                                         pseudo_seg=False)
TEST_DATASET = data_prep.PartNetDataset('./data/partnet/ins_seg_h5_for_detection/{}-{}/'.format(CAT, LEVEL),
                                        './data/partnet/ins_seg_h5_for_detection//cache/{}-{}/val_{}_00_cache.npz'.format(CAT, LEVEL, CONFIG.NUM_POINT),
                                        data_type='val', npoint=CONFIG.NUM_POINT, npoint_ins=CONFIG.NUM_POINT_INS, is_augment=False, permute_points=False,
                                        pseudo_seg=False)

CONFIG.NUM_GROUP = np.maximum(TRAIN_DATASET.ngroup, TEST_DATASET.ngroup)
TRAIN_DATASET.ngroup = CONFIG.NUM_GROUP
TEST_DATASET.ngroup = CONFIG.NUM_GROUP
CONFIG.NUM_CATEGORY = np.maximum(TRAIN_DATASET.nseg, TEST_DATASET.nseg)
TRAIN_DATASET.nseg = CONFIG.NUM_CATEGORY
TEST_DATASET.nseg = CONFIG.NUM_CATEGORY

def get_model(batch_size):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            CONFIG.BATCH_SIZE = batch_size
            pc_pl, pc_ins_pl, group_label_pl, group_indicator_pl, seg_label_pl, seg_label_per_group_pl, bbox_ins_pl = MODEL.placeholder_inputs(CONFIG)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            smpw_pl = tf.placeholder(tf.float32, shape=(CONFIG.BATCH_SIZE, CONFIG.NUM_POINT))
            end_points = MODEL.mrcnn_3d(pc_pl, pc_ins_pl, group_label_pl, group_indicator_pl, seg_label_pl, seg_label_per_group_pl, bbox_ins_pl, CONFIG, is_training_pl, mode='inference', bn_decay=None)
            # loss, end_points = MODEL.get_loss(end_points, CONFIG, alpha=1, smpw=smpw_pl, mode='inference')
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pc_pl': pc_pl,
               'pc_ins_pl': pc_ins_pl,
               'group_label_pl': group_label_pl,
               'group_indicator_pl': group_indicator_pl,
               'seg_label_pl': seg_label_pl,
               'seg_label_per_group_pl': seg_label_per_group_pl,
               'bbox_ins_pl': bbox_ins_pl,
               'smpw_pl': smpw_pl,
               'is_training_pl': is_training_pl,
               # 'loss': loss,
               'end_points': end_points}
        return sess, ops

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_pc = np.zeros((bsize, CONFIG.NUM_POINT, 3))
    batch_pc_ins = np.zeros((bsize, CONFIG.NUM_GROUP, CONFIG.NUM_POINT_INS, 3))
    batch_group_label = np.zeros((bsize, CONFIG.NUM_POINT), dtype=np.int32)
    batch_group_indicator = np.zeros((bsize, CONFIG.NUM_GROUP), dtype=np.int32)
    batch_seg_label = np.zeros((bsize, CONFIG.NUM_POINT), dtype=np.int32)
    batch_seg_label_per_group = np.zeros((bsize, CONFIG.NUM_GROUP), dtype=np.int32)
    batch_bbox_ins = np.zeros((bsize, CONFIG.NUM_GROUP, 6), dtype=np.float32)
    batch_smpw = np.ones((bsize, CONFIG.NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        pc, pc_ins, group_label, group_indicator, seg_label, seg_label_per_group, bbox_ins = dataset[idxs[i+start_idx]]
        batch_pc[i,...] = pc
        batch_pc_ins[i,...] = pc_ins
        batch_group_label[i,...] = group_label
        batch_group_indicator[i,...] = group_indicator
        batch_seg_label[i,...] = seg_label
        batch_seg_label_per_group[i,:] = seg_label_per_group
        batch_bbox_ins[i,...] = bbox_ins

    cat_count = np.zeros(CONFIG.NUM_CATEGORY)
    for i in range(1, CONFIG.NUM_CATEGORY):
        cat_count[i] += np.sum(batch_seg_label==i)
    total_cat_count = np.sum(cat_count).astype(np.float32)
    cat_count = np.multiply(1-np.divide(cat_count, total_cat_count+1e-8), cat_count>0)
    batch_smpw = cat_count[batch_seg_label.reshape(-1)].reshape((bsize, CONFIG.NUM_POINT)).astype(np.float32)

    return batch_pc, batch_pc_ins, batch_group_label, batch_group_indicator, batch_seg_label, batch_seg_label_per_group, batch_bbox_ins, batch_smpw


def soft_iou_normalized(pred1, pred2):
    # pred1: [NR, NP]
    # pred2: [NR, NP]
    intersection = np.matmul(pred1, np.transpose(pred2))
    union = np.sum(pred1,1,keepdims=True)+np.transpose(np.sum(pred2,1,keepdims=True))-intersection
    iou = np.divide(intersection, union+1e-6)
    N = np.diag(np.divide(1, np.sqrt(np.diag(iou))+1e-6))
    iou = np.matmul(N, np.matmul(iou, N))
    return iou


#### refine box whether use three scores: currently using 1
#### whether enlarge the box a bit in apply_delta: currently no
#### whether use conf to determine box: currently no
#### whether sort after get the detection: currently no
#### whether cut threshold after detection: currently no

def output_prediction_partnet(sess, ops, input_path, output_path, datatype='val'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    chunkfile_list = glob.glob(os.path.join(input_path, datatype+'-*.h5'))
    nchunk = len(chunkfile_list)
    sem_conf_th = 1e-4
    fb_conf_th = 0.01
    mask_th = 0.4
    for i in range(nchunk):
        hf = h5py.File(chunkfile_list[i], 'r')
        pc_chunk = hf['pts'].value #[B, N, 3]
        nfile = pc_chunk.shape[0]
        hf.close()
        with open(chunkfile_list[i].replace('.h5', '.json'), 'r') as fin:
            record = json.load(fin)
        output_mask = np.zeros((nfile, CONFIG.DETECTION_MAX_INSTANCES, CONFIG.NUM_POINT), dtype=np.bool)
        output_label = np.zeros((nfile, CONFIG.DETECTION_MAX_INSTANCES), dtype=np.uint8)
        output_valid = np.zeros((nfile, CONFIG.DETECTION_MAX_INSTANCES), dtype=np.bool)
        output_conf = np.zeros((nfile, CONFIG.DETECTION_MAX_INSTANCES), dtype=np.float32)
        for kk in range(nfile):
            print(i, kk/float(nfile))
            batch_pc = pc_chunk[kk:kk+1,:,:].astype(np.float32)
            batch_record = record[kk]
            batch_pc_ins = np.random.random((1, CONFIG.NUM_GROUP, CONFIG.NUM_POINT_INS, 3))
            batch_group_label = np.zeros((1, CONFIG.NUM_POINT), dtype=np.int32)
            batch_group_indicator = np.zeros((1, CONFIG.NUM_GROUP), dtype=np.int32)
            batch_seg_label = np.zeros((1, CONFIG.NUM_POINT), dtype=np.int32)
            batch_seg_label_per_group = np.zeros((1, CONFIG.NUM_GROUP), dtype=np.int32)
            batch_bbox_ins = np.random.random((1, CONFIG.NUM_GROUP, 6))
            batch_smpw = np.ones((1, CONFIG.NUM_POINT), dtype=np.float32)
            batch_group_indicator[0,1] = 1
            batch_seg_label[0,1] = 1
            batch_seg_label_per_group[0,1] = 1
            batch_group_label[0,1] = 1
            feed_dict = {ops['pc_pl']: batch_pc,
                         ops['pc_ins_pl']: batch_pc_ins,
                         ops['group_label_pl']: batch_group_label,
                         ops['group_indicator_pl']: batch_group_indicator,
                         ops['seg_label_pl']: batch_seg_label,
                         ops['seg_label_per_group_pl']: batch_seg_label_per_group,
                         ops['bbox_ins_pl']: batch_bbox_ins,
                         ops['smpw_pl']: batch_smpw,
                         ops['is_training_pl']: False}
            # mrcnn3d_mask_selected, pc_coord_cropped_final_unnormalized, detections, sem_class_logits, pc_seed, fb_prob = sess.run([ops['end_points']['mrcnn3d_mask_selected'],
            #     ops['end_points']['pc_coord_cropped_final_unnormalized'], ops['end_points']['detections'],
            #     ops['end_points']['sem_class_logits'], ops['end_points']['pc_seed'], ops['end_points']['fb_prob']], feed_dict=feed_dict)
            mrcnn3d_mask_selected, pc_coord_cropped_final_unnormalized, detections, sem_class_logits, pc_seed, fb_prob, bbox_ins_pred = sess.run([ops['end_points']['mrcnn3d_mask_selected'],
                                                                                                                                   ops['end_points']['pc_coord_cropped_final_unnormalized'], ops['end_points']['detections'],
                                                                                                                                   ops['end_points']['sem_class_logits'], ops['end_points']['pc_seed'], ops['end_points']['fb_prob'],
                                                                                                                                   ops['end_points']['bbox_ins_pred']], feed_dict=feed_dict)
            batch_pc = np.squeeze(batch_pc, 0) # [NUM_POINT, 3]
            mrcnn3d_mask_selected = np.squeeze(mrcnn3d_mask_selected, 0) # [DETECTION_MAX_INSTANCES, NUM_POINT_PER_ROI]
            pc_coord_cropped_final_unnormalized = np.squeeze(pc_coord_cropped_final_unnormalized, 0) # [DETECTION_MAX_INSTANCES, NUM_POINT_PER_ROI, 3]
            detections = np.squeeze(detections, 0) # [DETECTION_MAX_INSTANCES, 6+2]
            mrcnn3d_mask_selected = mrcnn3d_mask_selected[detections[:,6]>0,:]
            pc_coord_cropped_final_unnormalized = pc_coord_cropped_final_unnormalized[detections[:,6]>0,:,:]
            detections = detections[detections[:,6]>0,:]

            group_label_pred = np.zeros((detections.shape[0], batch_pc.shape[0]))
            for j in range(detections.shape[0]):
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pc_coord_cropped_final_unnormalized[j,:,:])
                _, sidx = nbrs.kneighbors(batch_pc)
                sidx = np.reshape(sidx, -1)
                group_label_pred[j,:] = mrcnn3d_mask_selected[j,sidx]
                roi_mask = np.logical_and(batch_pc>=np.reshape(detections[j,:3]-detections[j,3:6]/2, [1,3]),
                    batch_pc<=np.reshape(detections[j,:3]+detections[j,3:6]/2, [1,3]))
                roi_mask = np.logical_and(np.logical_and(roi_mask[:,0], roi_mask[:,1]), roi_mask[:,2])
                roi_mask = roi_mask.astype(np.float32)
                group_label_pred[j,:] = np.multiply(group_label_pred[j,:], roi_mask)
            

            confidence_pred = copy.deepcopy(detections[:,7].astype(np.float32))
            # confidence_pred = np.ones_like(detections[:,7])

            group_label_pred = (group_label_pred>mask_th).astype(np.bool) #[NUM_INSTANCE, NUM_POINT]
            seg_label_pred = copy.deepcopy(detections[:,6]).astype(np.uint8) #[NUM_INSTANCE]

            nins_out = group_label_pred.shape[0]
            output_mask[kk, :nins_out, :] = group_label_pred
            output_label[kk, :nins_out] = seg_label_pred # test data label starts from 0
            output_valid[kk, :nins_out] = True
            output_conf[kk, :nins_out] = confidence_pred

            if kk < 8 and group_label_pred.shape[0] > 0:
                mask = output_mask[kk, :nins_out, :]
                label = output_label[kk, :nins_out]
                valid = output_valid[kk, :nins_out]
                conf = output_conf[kk, :nins_out]
                bbox = np.squeeze(bbox_ins_pred)[np.argsort(np.squeeze(fb_prob)[:, 1])[::-1][:nins_out]]
                gen_visu(os.path.join(output_path, 'visu'), kk, batch_pc, mask, valid, conf, bbox, batch_record)

        hf = h5py.File(os.path.join(output_path, os.path.basename(chunkfile_list[i])),'w')
        hf.create_dataset('mask', data=output_mask)
        hf.create_dataset('label', data=output_label)
        hf.create_dataset('valid', data=output_valid)
        hf.create_dataset('conf', data=output_conf)
        hf.close()


def export_label(out, label):
    with open(out, 'w') as fout:
        for i in range(label.shape[0]):
            fout.write('%d\n' % label[i])

def export_pts(out, v):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))

def export_bbox(out, v):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], v[i, 3], v[i, 4], v[i, 5]))

def render_pts_pptk(out, pts, delete_img=False, point_size=6, point_color='FF0000FF'):
    tmp_pts = out.replace('.png', '.pts')
    export_pts(tmp_pts, pts)

def render_pts_with_label_pptk(out, pts, label, delete_img=False, base=0, point_size=6):
    tmp_pts = out.replace('.png', '.pts')
    tmp_label = out.replace('.png', '.label')

    label += base

    export_pts(tmp_pts, pts)
    export_label(tmp_label, label)

def render_pts_with_bbox_pptk(out, pts, bbox, delete_img=False, base=0, point_size=6):
    tmp_pts = out.replace('.png', '.pts')
    tmp_bbox = out.replace('.png', '.bbox')

    export_pts(tmp_pts, pts)
    export_bbox(tmp_bbox, bbox)

def render_pts_with_mask_pptk(out, pts, mask, delete_img=False, base=0, point_size=6):
    tmp_pts = out.replace('.png', '.pts')
    tmp_label = out.replace('.png', '.mask')

    export_pts(tmp_pts, pts)
    export_label(tmp_label, mask)

def gen_visu(visu_dir, base_idx, pts, mask, valid, conf, bbox, record, num_pts_to_visu=1000):
    n_ins = mask.shape[0]

    pts_dir = os.path.join(visu_dir, 'pts')
    info_dir = os.path.join(visu_dir, 'info')
    child_dir = os.path.join(visu_dir, 'child')

    if base_idx == 0:
        os.makedirs(pts_dir)
        os.makedirs(info_dir)
        os.makedirs(child_dir)

    cur_pts = pts
    cur_mask = mask
    cur_valid = valid
    cur_conf = conf
    cur_bbox = bbox
    cur_record = record

    cur_idx_to_visu = np.arange(CONFIG.NUM_POINT)
    np.random.shuffle(cur_idx_to_visu)
    cur_idx_to_visu = cur_idx_to_visu[:num_pts_to_visu]

    cur_shape_prefix = 'shape-%03d' % (base_idx)
    out_fn = os.path.join(pts_dir, cur_shape_prefix+'.png')
    render_pts_with_mask_pptk(out_fn, cur_pts[cur_idx_to_visu], cur_mask[cur_valid, :].argmax(0)[cur_idx_to_visu])
    render_pts_with_bbox_pptk(out_fn, cur_pts[cur_idx_to_visu], cur_bbox)
    out_fn = os.path.join(info_dir, cur_shape_prefix+'.txt')
    with open(out_fn, 'w') as fout:
        fout.write('model_id: %s, anno_id: %s\n' % (cur_record['model_id'], cur_record['anno_id']))

    cur_child_dir = os.path.join(child_dir, cur_shape_prefix)
    os.mkdir(cur_child_dir)
    child_pred_dir = os.path.join(cur_child_dir, 'pred')
    os.mkdir(child_pred_dir)
    child_info_dir = os.path.join(cur_child_dir, 'info')
    os.mkdir(child_info_dir)

    cur_conf[~cur_valid] = 0.0
    idx = np.argsort(-cur_conf)
    for j in range(n_ins):
        cur_idx = idx[j]
        if cur_valid[cur_idx]:
            cur_part_prefix = 'part-%03d' % j
            out_fn = os.path.join(child_pred_dir, cur_part_prefix+'.png')
            render_pts_with_label_pptk(out_fn, cur_pts[cur_idx_to_visu], cur_mask[cur_idx, cur_idx_to_visu].astype(np.int32))
            out_fn = os.path.join(child_info_dir, cur_part_prefix+'.txt')
            with open(out_fn, 'w') as fout:
                fout.write('part idx: %d\n' % cur_idx)
                fout.write('score: %f\n' % cur_conf[cur_idx])
                fout.write('#pts: %d\n' % np.sum(cur_mask[cur_idx, :]))


if __name__ == '__main__':
    BATCH_SIZE = 1
    ##### Generate ground truth

    ##### Generate prediction
    sess, ops = get_model(batch_size=1)
    output_prediction_partnet(sess, ops, './data/partnet/ins_seg_h5_gt/{}-{}/'.format(CAT, LEVEL), os.path.join(LOG_DIR, 'pred', '{}-{}'.format(CAT, LEVEL)), datatype='test')

