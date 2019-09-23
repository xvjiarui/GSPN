import argparse
import os
import sys
from collections import defaultdict
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from eval_utils import eval_recall_iou_nosem_fusion
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', type=str, help='log prediction directory [default: log]')
parser.add_argument('--gt_dir', type=str, help='log prediction directory [default: log]')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU Threshold [default: 0.5]')
parser.add_argument('--plot_dir', type=str, default=None, help='PR Curve Plot Output Directory [default: None, meaning no output]')
FLAGS = parser.parse_args()



gt_dir = FLAGS.gt_dir
pred_dir = FLAGS.pred_dir
gt_folders = os.listdir(gt_dir)
result_table = defaultdict(dict)
result_file = os.path.join(pred_dir, 'result_table.pkl')
if os.path.exists(result_file):
    with open(result_file, 'wb') as f:
        result_table = pickle.load(f)
else:
    for gt_folder in gt_folders:
        category, level = gt_folder.split('-')
        if category in ['Chair', 'Table']:
            continue
        gt_in_dir = os.path.join(gt_dir, gt_folder)
        recalls = eval_recall_iou_nosem_fusion(gt_in_dir, os.path.join(pred_dir, category), iou_threshold=FLAGS.iou_threshold, plot_dir=FLAGS.plot_dir)
        result_table[category][level] = dict()
        result_table[category][level]['recalls'] = recalls
        result_table[category][level]['avg_recall'] = np.mean(recalls)
        print('{}: {}'.format(gt_folder, np.mean(recalls)))

    with open(result_file, 'wb') as f:
        pickle.dump(result_table, f)
for cate in result_table.keys():
    for level in result_table[cate].keys():
        recall = result_table[cate][level]['recalls']
        avg_recall = result_table[cate][level]['avg_recall']
        print('{}-{}: {}'.format(cate, level, avg_recall))

print 'Done'

