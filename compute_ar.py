import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from eval_utils import eval_recall_iou_nosem_fusion
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, help='Category name [default: Chair]')
parser.add_argument('--level_id', type=int, nargs='+', help='Level ID [default: 3]')
parser.add_argument('--pred_dir', type=str, help='log prediction directory [default: log]')
parser.add_argument('--gt_dir', type=str, help='log prediction directory [default: log]')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU Threshold [default: 0.5]')
parser.add_argument('--plot_dir', type=str, default=None, help='PR Curve Plot Output Directory [default: None, meaning no output]')
FLAGS = parser.parse_args()


gt_dir = FLAGS.gt_dir
pred_dir = FLAGS.pred_dir
for level in FLAGS.level_id:
    gt_in_dir = os.path.join(gt_dir, '{}-{}'.format(FLAGS.category, level))
    recalls = eval_recall_iou_nosem_fusion(gt_in_dir, os.path.join(pred_dir, FLAGS.category), iou_threshold=FLAGS.iou_threshold, plot_dir=FLAGS.plot_dir)
    print(recalls)
    print('mRecall %f'%np.mean(recalls))

