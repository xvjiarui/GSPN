import os
import sys
import numpy as np
import h5py
import copy
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
import tf_sampling
import tensorflow as tf
import scipy.io as sio
from scipy import stats
import glob

class PartNetDataset():
    def __init__(self, src_data_path, cache_path, data_type, npoint=10000, npoint_ins=512, is_augment=False, permute_points=True, pseudo_seg=False):
        self.npoint = npoint
        self.npoint_ins = npoint_ins
        self.ngroup = 0
        self.nseg = 0
        self.is_augment = is_augment
        self.permute_points = permute_points
        self.data_type = data_type # train, val, test
        self.data_list = {}
        self.pseudo_seg = pseudo_seg
        if os.path.exists(cache_path):
            self.data_list = np.load(cache_path, allow_pickle=True)['data_list'].item()
            self.ngroup = np.load(cache_path, allow_pickle=True)['ngroup'].item()
            self.nseg = np.load(cache_path, allow_pickle=True)['nseg'].item()
        else:
            self.cache_file(src_data_path, cache_path)

    def cache_file(self, src_data_path, cache_path):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.device('/gpu:0'):
            pc_tf = tf.placeholder(tf.float32)
            ind_tf = tf_sampling.farthest_point_sample(self.npoint, pc_tf)
            ind_ins_tf = tf_sampling.farthest_point_sample(self.npoint_ins, pc_tf)
        sess = tf.Session(config=config)

        chunk_file = []
        if isinstance(src_data_path, list):
            for src in src_data_path:
                chunk_file.extend(glob.glob(os.path.join(src, self.data_type+'*.h5')))
        else:
            chunk_file.extend(glob.glob(os.path.join(src_data_path, self.data_type+'*.h5')))
        # chunk_file = glob.glob(os.path.join(src_data_path, self.data_type+'*-00.h5'))
        print chunk_file
        chunk_file.sort()
        data_index = 0
        for i, cur_chunk in enumerate(chunk_file):
            hf = h5py.File(cur_chunk, 'r')
            pc_chunk = hf['pts'].value #[B, N, 3]
            seg_chunk = hf['gt_label'].value # [B, N], (0~curnseg), 0 is background points
            group_chunk = np.concatenate((np.expand_dims(hf['gt_other_mask'].value, 1),
                hf['gt_mask'].value), 1) # [B, 201, N], binary mask
            group_chunk = np.argmax(group_chunk, 1) # [B, N], (0~curngroup), 0 is background
            nfile = pc_chunk.shape[0]
            hf.close()
            #### update ngroup and seg
            if np.max(group_chunk)+1>self.ngroup:
                self.ngroup = np.max(group_chunk)+1
            if np.max(seg_chunk)+1>self.nseg:
                self.nseg = np.max(seg_chunk)+1
            for index in range(nfile):
                print(i, np.float32(index)/nfile)
                curpc = pc_chunk[index,:,:].astype(np.float32) # [npoint, 3]
                curgroup = group_chunk[index,:].astype(np.int32) # group zero is background, [npoint]
                curseg = seg_chunk[index,:].astype(np.int32) # 0 is background, [npoint]
                curngroup = np.max(curgroup)+1 # group zero is background, [ngroup+1]
                #### sample instance for each group
                pc_ins = np.zeros((curngroup, self.npoint_ins, 3), dtype=np.float32)
                for j in range(1,curngroup):
                    if np.sum(curgroup==j)<5:
                        continue
                    curins = curpc[curgroup==j,:]
                    if self.npoint_ins<curins.shape[0]:
                        choice = sess.run(ind_ins_tf, feed_dict={pc_tf: np.expand_dims(curins,0)})[0]
                        pc_ins[j,:,:] = curins[choice,:]
                    elif self.npoint_ins==curins.shape[0]:
                        pc_ins[j,:,:] = copy.deepcopy(curins)
                    else:
                        choice = np.random.choice(curins.shape[0], self.npoint_ins - curins.shape[0])
                        pc_ins[j,:,:] = np.concatenate((curins,curins[choice,:]), 0)
                #### remove group less than 5 points
                valid_group_indicator = np.ones(curngroup-1)
                target_group_idx = np.zeros(curngroup)
                count = 0
                for j in range(1,curngroup):
                    if np.sum(curgroup==j)<5:
                        valid_group_indicator[j-1] = 0
                    else:
                        count += 1
                        target_group_idx[j] = count
                curgroup = self.changem(curgroup, np.arange(curngroup).astype(np.int32), target_group_idx).astype('int32')
                valid_group_indicator = np.concatenate(([1], valid_group_indicator))
                pc_ins = pc_ins[valid_group_indicator==1,:,:]
                curngroup = count+1 # group zero is background, [ngroup+1]

                pc = curpc
                group_label = curgroup
                seg_label = curseg
                seg_label_per_group = np.zeros(curngroup).astype(np.int32)
                for j in range(1,curngroup):
                    seg_label_per_group[j] = stats.mode(seg_label[group_label==j])[0][0]
                self.data_list[data_index] = (pc, group_label, seg_label, seg_label_per_group, pc_ins, curngroup)
                data_index += 1

        np.savez_compressed(cache_path, data_list=self.data_list, ngroup=self.ngroup, nseg=self.nseg)

    def gen_rotation_matrix(self):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        return rotation_matrix

    def gen_small_rotation_matrix(self, range):
        rotation_angle = (2*np.random.uniform()-1) * range
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        return rotation_matrix

    def changem(self, input_array, source_idx, target_idx):
        mapping = {}
        for i, sidx in enumerate(source_idx):
            mapping[sidx] = target_idx[i]
        input_array = np.array([mapping[i] for i in input_array])
        return input_array

    def __getitem__(self, index):
        '''
        Return:
            pc: [npoint, 3], world coord sys
            pc_ins_full: [ngroup, npoint_ins, 3], world coord sys, all zero for background ins
            group_label: [npoint], 0 means background
            group_indicator: [ngroup], indicates which group is valid, usually the first few
            seg_label: [npoint], 0 means background class
        '''
        pc, group_label, seg_label, seg_label_per_group, pc_ins, curngroup = self.data_list[index]
        if self.permute_points:
            ridx = np.random.permutation(pc.shape[0])
            pc = copy.deepcopy(pc[ridx,:])
            seg_label = copy.deepcopy(seg_label[ridx])
            group_label = copy.deepcopy(group_label[ridx])
        else:
            pc = copy.deepcopy(pc)
            seg_label = copy.deepcopy(seg_label)
            group_label = copy.deepcopy(group_label)

        seg_label_per_group_full = np.zeros(self.ngroup, dtype=np.int32)
        seg_label_per_group_full[:curngroup] = copy.deepcopy(seg_label_per_group)
        group_indicator = np.zeros((self.ngroup), dtype=np.int32)
        group_indicator[:curngroup] = 1
        pc_ins_full = np.zeros((self.ngroup, self.npoint_ins, 3), dtype=np.float32)
        pc_ins_full[:curngroup,:,:] = pc_ins
        if self.is_augment:
            # R = self.gen_rotation_matrix()
            # R = self.gen_small_rotation_matrix(np.pi/18)
            # pc = np.matmul(pc, R)
            # pc_ins_full = np.reshape(np.matmul(np.reshape(pc_ins_full, [-1, 3]), R), pc_ins_full.shape)
            # aug translation
            # t = np.random.normal(0,0.1,[1,3])
            # pc += t
            # pc_ins_full[:curngroup,:,:] += np.reshape(t, [1,1,3])
            # jittering
            sigma = 0.01
            clip = 0.05
            jittered_data = np.clip(sigma * np.random.randn(pc.shape[0], pc.shape[1]), -1*clip, clip)
            pc += jittered_data

        bbox_ins_full = np.zeros((self.ngroup, 6), dtype=np.float32)
        bbox_ins_full[:curngroup, :3] = (np.max(pc_ins_full[:curngroup,:,:],1)+np.min(pc_ins_full[:curngroup,:,:],1))/2
        bbox_ins_full[:curngroup, 3:] = np.max(pc_ins_full[:curngroup,:,:],1)-np.min(pc_ins_full[:curngroup,:,:],1)
        if self.pseudo_seg:
            seg_label[:] = 1
            seg_label_per_group_full[:] = 1
        return pc, pc_ins_full, group_label, group_indicator, seg_label, seg_label_per_group_full, bbox_ins_full


    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    #### PartNet
    # d = PartNetDataset('/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/Chair-3/','/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/Chair-3/val_10000_cache.npz', data_type='val', npoint=10000, npoint_ins=512, is_augment=False, permute_points=False)
    # print(len(d))
    # d = PartNetDataset('/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/Chair-3/','/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/Chair-3/train_10000_cache.npz', data_type='train', npoint=10000, npoint_ins=512, is_augment=True, permute_points=True)
    # print(len(d))
    # d = PartNetDataset('/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/Table-3/','/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/Table-3/val_10000_cache.npz', data_type='val', npoint=10000, npoint_ins=512, is_augment=False, permute_points=False)
    # print(len(d))
    # d = PartNetDataset('/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/Table-3/','/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/Table-3/train_10000_cache.npz', data_type='train', npoint=10000, npoint_ins=512, is_augment=True, permute_points=True)
    # print(len(d))
    # d = PartNetDataset('/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/Lamp-3/','/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/Lamp-3/val_10000_cache.npz', data_type='val', npoint=10000, npoint_ins=512, is_augment=False, permute_points=False)
    # print(len(d))
    # d = PartNetDataset('/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/Lamp-3/','/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/Lamp-3/train_10000_cache.npz', data_type='train', npoint=10000, npoint_ins=512, is_augment=True, permute_points=True)
    # print(len(d))
    # d = PartNetDataset('/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/StorageFurniture-3/','/orionp2/indoorinstseg/Data/PartNet/ins_seg_h5_for_detection/StorageFurniture-3/val_10000_cache.npz', data_type='val', npoint=10000, npoint_ins=512, is_augment=False, permute_points=False)
    # print(len(d))
    d = PartNetDataset('./data/partnet/ins_seg_h5_for_detection/StorageFurniture-3/','./data/partnet/ins_seg_h5_for_detection/StorageFurniture-3/train_10000_cache.npz', data_type='train', npoint=10000, npoint_ins=512, is_augment=True, permute_points=True)
    print(len(d))
    pc, pc_ins_full, group_label, group_indicator, seg_label, seg_label_per_group, bbox_ins_full = d[0]

    import pdb
    pdb.set_trace()

    # out = {}
    # out['pc'] = pc
    # out['pc_ins_full'] = pc_ins_full
    # out['group_label'] = group_label
    # out['group_indicator'] = group_indicator
    # out['seg_label'] = seg_label
    # out['seg_label_per_group'] = seg_label_per_group
    # out['bbox_ins_full'] = bbox_ins_full
    # sio.savemat('/orions4-zfs/projects/ericyi/InsSeg/Code/matlab/tmp_partnet.mat',out)

    # #### ScanNet
    # d = ScanNetDataset('/orions2-zfs/projects/zhaow/scannet_color', '/orionp2/indoorinstseg/Data/ScanNetV2/scannet_val.txt','/orionp2/indoorinstseg/Data/ScanNetV2/test_18000_cache.npz', npoint=18000, npoint_ins=512, is_augment=False)
    # print(len(d))
    # d = ScanNetDataset('/orions2-zfs/projects/zhaow/scannet_color', '/orionp2/indoorinstseg/Data/ScanNetV2/scannet_train.txt','/orionp2/indoorinstseg/Data/ScanNetV2/train_18000_cache.npz', npoint=18000, npoint_ins=512, is_augment=True)
    # print(len(d))
    # pc, color, pc_ins_full, group_label, group_indicator, seg_label, bbox_ins_full = d[0]

    # out = {}
    # out['pc'] = pc
    # out['color'] = color
    # out['pc_ins_full'] = pc_ins_full
    # out['group_label'] = group_label
    # out['group_indicator'] = group_indicator
    # out['seg_label'] = seg_label
    # out['bbox_ins_full'] = bbox_ins_full
    # sio.savemat('/orions4-zfs/projects/ericyi/InsSeg/Code/matlab/tmp_mrcnn.mat',out)

    # #### Old
    # sys.path.append('utils')
    # # import show3d_balls
    # # d = SynDataset('syndata/train.txt', 'syndata/train_2048_cache.npz', npoint=2048, npoint_ins=512, is_augment=True)
    # d = SynDataset('syndata/test.txt', 'syndata/test_2048_cache.npz', npoint=2048, npoint_ins=512, is_augment=True)
    # print(len(d))
    # # print(d.ngroup, d.file_list)
    # pc, pc_ins_full, group_label, group_indicator, seg_label, bbox_ins_full = d[0]

    # # color = np.ones((2048,3))
    # # color[:,0] = group_label/np.max(group_label).astype('float32')
    # print(np.unique(group_label))
    # # show3d_balls.showpoints(pc, ballradius=8, c_gt=color)

    # # color = np.ones((2048,3))
    # # color[:,0] = seg_label/np.max(seg_label).astype('float32')
    # print(np.unique(seg_label))
    # # show3d_balls.showpoints(pc, ballradius=8, c_gt=color)

    # print(group_indicator)
    # print(pc_ins_full.shape)

    # show3d_balls.showpoints(pc_ins_full[0,:,:], ballradius=8)
    # show3d_balls.showpoints(pc_ins_full[1,:,:], ballradius=8)
    # show3d_balls.showpoints(pc_ins_full[2,:,:], ballradius=8)
    # show3d_balls.showpoints(pc_ins_full[3,:,:], ballradius=8)
    # pc, pc_ins_full, group_label, group_indicator, seg_label = d[1]
    # show3d_balls.showpoints(pc, ballradius=8)
    # show3d_balls.showpoints(pc_ins_full[0,:,:], ballradius=8)
    # show3d_balls.showpoints(pc_ins_full[1,:,:], ballradius=8)
    # show3d_balls.showpoints(pc_ins_full[2,:,:], ballradius=8)
    # show3d_balls.showpoints(pc_ins_full[3,:,:], ballradius=8)
