import os
import sys
import time
import torch
import random
import tifffile
import numpy as np


from .consistency_aug_perturbations import Intensity
from .consistency_aug_perturbations import GaussBlur
from .consistency_aug_perturbations import GaussNoise
from .consistency_aug_perturbations import Cutout
from connectomics.data.augmentation import Grayscale
from connectomics.data.augmentation import Rotate
from connectomics.data.augmentation import Flip
from connectomics.data.augmentation import Elastic
from connectomics.data.augmentation import Rescale
from connectomics.data.utils.data_io import readh5
from torch.utils.data import Dataset
from connectomics.data.utils.data_weight import weight_binary_ratio

class Superhuman_aug(Dataset):
    def __init__(self, cfg):
        super(Superhuman_aug, self).__init__()

        self.cfg = cfg
        self.crop_size = list(self.cfg.MODEL.INPUT_SIZE)
        self.out_size = self.crop_size

        self.reject_p = 0.95
        self.reject_size_thres = 100
        self.background = 0
        
        self.train_path = self.cfg.DATASET.INPUT_PATH
        self.test_path = self.cfg.DATASET.INPUT_PATH
        self.train_set = list(self.cfg.DATASET.IMAGE_NAME)
        self.train_set_label = list(self.cfg.DATASET.LABEL_NAME)

        self.test_set = list(self.cfg.DATASET.UNLABLE_IMAGE_NAME)
        self.num_vol = len(self.train_set)
        self.num_vol_test = len(self.test_set)

        self.datasets = []
        self.labels_pre = []
        self.labels_post = []
        for idx, data in enumerate(self.train_set):
            print('load source data:' + data + ' ...')
            raw = tifffile.imread(os.path.join(self.train_path, self.train_set[idx]))
            lb = tifffile.imread(os.path.join(self.train_path, self.train_set_label[idx]))
            lb_pre = (lb==1).astype(lb.dtype)
            lb_post = (lb==2).astype(lb.dtype)
            self.datasets.append(raw)
            self.labels_pre.append(lb_pre)
            self.labels_post.append(lb_post)

        self.test_datasets = []
        for data in self.test_set:
            print('load target data:' + data + ' ...')
            raw = readh5(os.path.join(self.test_path, data))
            self.test_datasets.append(raw)

        # augmentation initoalization
        # augmentation
        self.if_filp_aug = True
        self.if_rotation_aug = False
        self.if_scale_aug = False
        self.if_elastic_aug = False
        self.if_intensity_aug = True
        self.if_noise_aug = True
        self.if_blur_aug = True
        self.if_mask_aug = True

        # augmentation for unlabeled data
        self.if_intensity_aug_unlabel = True
        self.if_noise_aug_unlabel = True
        self.min_noise_std = 0.01
        self.max_noise_std = 0.2
        self.if_mask_aug_unlabel = True
        self.if_blur_aug_unlabel = True
        self.min_kernel_size = 3
        self.max_kernel_size = 9
        self.min_sigma = 0
        self.max_sigma = 2

        # padding for random rotation
        self.padding = 50
        if self.if_rotation_aug:
            self.crop_from_origin = [self.crop_size[x] + 2*self.padding for x in range(3)]
        else:
            self.crop_from_origin = self.crop_size

        self.augs_init()
        self.perturbations_init()

    def __len__(self):
        return int(sys.maxsize)

    def _is_fg(self, out_label: np.ndarray) -> bool:
        """Decide whether the sample belongs to a foreground decided
        by the rejection sampling criterion.
        """
        
        p = self.reject_p
        size_thres = self.reject_size_thres
        if size_thres > 0:
            temp = out_label.copy().astype(int)
            temp = (temp != self.background).astype(int).sum()
            if temp < size_thres and random.random() < p:
                return False

        return True
    
    def __getitem__(self, index):
        # for source data
        k = random.randint(0, self.num_vol-1)
        used_data = self.datasets[k]
        used_label_pre = self.labels_pre[k]
        used_label_post = self.labels_post[k]

        raw_data_shape = used_data.shape
        while True:
            random_z = random.randint(0, raw_data_shape[0]-self.crop_from_origin[0])
            random_y = random.randint(0, raw_data_shape[1]-self.crop_from_origin[1])
            random_x = random.randint(0, raw_data_shape[2]-self.crop_from_origin[2])
            imgs_src = used_data[random_z:random_z+self.crop_from_origin[0], \
                                random_y:random_y+self.crop_from_origin[1], \
                                random_x:random_x+self.crop_from_origin[2]]
            lb_pre = used_label_pre[random_z:random_z+self.crop_from_origin[0], \
                            random_y:random_y+self.crop_from_origin[1], \
                            random_x:random_x+self.crop_from_origin[2]]
            lb_post = used_label_post[random_z:random_z+self.crop_from_origin[0], \
                            random_y:random_y+self.crop_from_origin[1], \
                            random_x:random_x+self.crop_from_origin[2]]
            imgs_src = imgs_src.astype(np.float32) / 255.0
            data = {'image': imgs_src, 'label1': lb_pre, 'label2': lb_post}
            if np.random.rand() < 0.5:
                data = self.augs_mix(data)
            imgs_src = data['image']
            lb_pre = data['label1']
            lb_post = data['label2']
            if self.if_rotation_aug:
                imgs_src = center_crop(imgs_src, det_shape=self.crop_size)
                lb_pre = center_crop(lb_pre, det_shape=self.crop_size)
                lb_post = center_crop(lb_post, det_shape=self.crop_size)
            if self._is_fg(lb_pre + lb_post):
                break
        # for target data
        k = random.randint(0, self.num_vol_test-1)
        used_data = self.test_datasets[k]

        raw_data_shape = used_data.shape
        random_z = random.randint(0, raw_data_shape[0]-self.crop_from_origin[0])
        random_y = random.randint(0, raw_data_shape[1]-self.crop_from_origin[1])
        random_x = random.randint(0, raw_data_shape[2]-self.crop_from_origin[2])
        imgs_tgt = used_data[random_z:random_z+self.crop_from_origin[0], \
                            random_y:random_y+self.crop_from_origin[1], \
                            random_x:random_x+self.crop_from_origin[2]]
        imgs_tgt = imgs_tgt.astype(np.float32) / 255.0
        
        if self.if_rotation_aug:
            imgs_tgt = center_crop(imgs_tgt, det_shape=self.crop_size)
            
        imgs_tgt_aug = self.apply_perturbations(imgs_tgt.copy())
        
        imgs_src = imgs_src[np.newaxis, ...]
        imgs_tgt_aug = imgs_tgt_aug[np.newaxis, ...]
        imgs_tgt = imgs_tgt[np.newaxis, ...]
        lb_pre = lb_pre[np.newaxis, ...]
        lb_post = lb_post[np.newaxis, ...]

        imgs = np.concatenate([imgs_src, imgs_tgt_aug, imgs_tgt], axis=0)
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        lb_pre = np.ascontiguousarray(lb_pre, dtype=np.float32)
        lb_post = np.ascontiguousarray(lb_post, dtype=np.float32)

        pre_weight = weight_binary_ratio(lb_pre)
        post_weight = weight_binary_ratio(lb_post)

        return imgs, np.concatenate([lb_pre, pre_weight]), np.concatenate([lb_post, post_weight])

    def augs_init(self):
        # https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
        self.aug_rotation = Rotate(p=0.5)
        self.aug_rescale = Rescale(p=0.5)
        self.aug_flip = Flip(p=1.0, do_ztrans=0)
        self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0)
        self.aug_grayscale = Grayscale(p=0.75)
        self.aug_gaussnoise = GaussNoise(min_std=self.min_noise_std, max_std=self.max_noise_std, norm_mode='trunc')
        self.aug_gaussblur = GaussBlur(min_kernel=self.min_kernel_size, max_kernel=self.max_kernel_size, min_sigma=self.min_sigma, max_sigma=self.max_sigma)
        self.aug_cutout = Cutout()

    def perturbations_init(self):
        self.per_intensity = Intensity()
        self.per_gaussnoise = GaussNoise(min_std=self.min_noise_std, max_std=self.max_noise_std, norm_mode='trunc')
        self.per_gaussblur = GaussBlur(min_kernel=self.min_kernel_size, max_kernel=self.max_kernel_size, min_sigma=self.min_sigma, max_sigma=self.max_sigma)
        self.per_cutout = Cutout()

    def augs_mix(self, data):
        if self.if_filp_aug and random.random() > 0.5:
            data = self.aug_flip(data)
        if self.if_rotation_aug and random.random() > 0.5:
            data = self.aug_rotation(data)
        if self.if_scale_aug and random.random() > 0.5:
            data = self.aug_rescale(data)
        if self.if_elastic_aug and random.random() > 0.5:
            data = self.aug_elastic(data)
        if self.if_intensity_aug and random.random() > 0.5:
            data = self.aug_grayscale(data)
        if self.if_noise_aug and random.random() > 0.5:
            data['image'] = self.aug_gaussnoise(data['image'])
        if self.if_blur_aug and random.random() > 0.5:
            data['image'] = self.aug_gaussblur(data['image'])
        if self.if_mask_aug and random.random() > 0.5:
            data['image'] = self.aug_cutout(data['image'])
        return data

    def apply_perturbations(self, data):
        all_pers = [self.if_intensity_aug_unlabel, self.if_noise_aug_unlabel, self.if_blur_aug_unlabel, self.if_mask_aug_unlabel]
        # select used perturbations
        used_pers = []
        for k, value in enumerate(all_pers):
            if value:
                used_pers.append(k)
        # select which one perturbation to use
        if len(used_pers) == 0:
            # do nothing
            raise AttributeError('Must be one augmentation!')
        elif len(used_pers) == 1:
            # No choise if only one perturbation can be used
            rand_per = used_pers[0]
        else:
            rand_per = random.choice(used_pers)
        # do augmentation
        # intensity
        if rand_per == 0:
            data = self.per_intensity(data)
        # noise
        if rand_per == 1:
            data = self.per_gaussnoise(data)
        # blur
        if rand_per == 2:
            data = self.per_gaussblur(data)
        # mask or cutout
        if rand_per == 3:
            data = self.per_cutout(data)
        return data
    
def center_crop(image, det_shape=[160, 160, 160]):
    # To prevent overflow
    # image = np.pad(image, ((20,20),(20,20),(20,20)), mode='reflect')
    src_shape = image.shape
    shift0 = (src_shape[0] - det_shape[0]) // 2
    shift1 = (src_shape[1] - det_shape[1]) // 2
    shift2 = (src_shape[2] - det_shape[2]) // 2
    assert shift0 > 0 or shift1 > 0 or shift2 > 0, "overflow in center-crop"
    image = image[shift0:shift0+det_shape[0], shift1:shift1+det_shape[1], shift2:shift2+det_shape[2]]
    return image

import glob
import h5py
# class Superhuman_aug_test(Dataset):
#     def __init__(self, cfg, data_name, mode='valid'):
#         # basic settings
#         self.cfg = cfg
#         self.data_name = data_name
#         self.mode = mode
#         self.crop_size = list(self.cfg.DATA.crop_size)
#         self.out_size = self.crop_size
#         self.stride = [self.crop_size[x] // 2 for x in range(3)]

#         self.test_set = list(self.cfg.DATASET.UNLABLE_IMAGE_NAME)

#         self.test_datasets = []
#         for data in self.test_set:
#             print('load target data:' + data + ' ...')
#             raw = readh5(os.path.join(self.test_path, data))
#             self.test_datasets.append(raw)

#         self.labels_pre = np.zeros_like(self.datasets, dtype=np.float32)
#         self.labels_post = np.zeros_like(self.datasets, dtype=np.float32)

#         if self.valid_split is not None:
#             self.datasets = self.datasets[-self.valid_split:, ...]
#             self.labels_pre = self.labels_pre[-self.valid_split:, ...]
#             self.labels_post = self.labels_post[-self.valid_split:, ...]
 
#         print('Dataset size:', self.datasets.shape)
#         size_z, size_y, size_x = self.datasets.shape
#         pad_z, num_z = pad_num(size_z, self.crop_size[0])
#         pad_y, num_y = pad_num(size_y, self.crop_size[1])
#         pad_x, num_x = pad_num(size_x, self.crop_size[2])

#         self.padding_zyx = [pad_z, pad_y, pad_x]
#         self.num_zyx = [num_z, num_y, num_x]
#         self.iters_num = num_z * num_y * num_x
#         print('padding size:', self.padding_zyx)
#         print('iter num:', self.num_zyx)
#         self.labels_pre_origin = self.labels_pre.copy()
#         self.labels_post_origin = self.labels_post.copy()

#         self.datasets = np.pad(self.datasets, ((self.padding_zyx[0], self.padding_zyx[0]), \
#                                             (self.padding_zyx[1], self.padding_zyx[1]), \
#                                             (self.padding_zyx[2], self.padding_zyx[2])), mode='reflect')
#         self.labels_pre = np.pad(self.labels_pre, ((self.padding_zyx[0], self.padding_zyx[0]), \
#                                             (self.padding_zyx[1], self.padding_zyx[1]), \
#                                             (self.padding_zyx[2], self.padding_zyx[2])), mode='reflect')
#         self.labels_post = np.pad(self.labels_post, ((self.padding_zyx[0], self.padding_zyx[0]), \
#                                             (self.padding_zyx[1], self.padding_zyx[1]), \
#                                             (self.padding_zyx[2], self.padding_zyx[2])), mode='reflect')

#         self.raw_data_shape = list(self.datasets.shape)
#         self.reset_output()
#         self.weight_vol = self.get_weight()

#     def __getitem__(self, index):
#         # pos_data = index // self.iters_num
#         # pre_data = index % self.iters_num
#         pos_z = index // (self.num_zyx[1] * self.num_zyx[2])
#         pos_xy = index % (self.num_zyx[1] * self.num_zyx[2])
#         pos_x = pos_xy // self.num_zyx[2]
#         pos_y = pos_xy % self.num_zyx[2]

#         # find position
#         fromz = pos_z * self.stride[0]
#         endz = fromz + self.crop_size[0]
#         if endz > self.raw_data_shape[0]:
#             endz = self.raw_data_shape[0]
#             fromz = endz - self.crop_size[0]
#         fromy = pos_y * self.stride[1]
#         endy = fromy + self.crop_size[1]
#         if endy > self.raw_data_shape[1]:
#             endy = self.raw_data_shape[1]
#             fromy = endy - self.crop_size[1]
#         fromx = pos_x * self.stride[2]
#         endx = fromx + self.crop_size[2]
#         if endx > self.raw_data_shape[2]:
#             endx = self.raw_data_shape[2]
#             fromx = endx - self.crop_size[2]
#         self.pos = [fromz, fromy, fromx]

#         imgs = self.datasets[fromz:endz, fromy:endy, fromx:endx]
#         lb_pre = self.labels_pre[fromz:endz, fromy:endy, fromx:endx]
#         lb_post = self.labels_post[fromz:endz, fromy:endy, fromx:endx]
#         # weightmap = weight_binary_ratio(lb)

#         imgs = imgs.astype(np.float32) / 255.0
#         imgs = imgs[np.newaxis, ...]
#         lb_pre = lb_pre[np.newaxis, ...]
#         lb_post = lb_post[np.newaxis, ...]
#         # weightmap = weightmap[np.newaxis, ...]
#         imgs = np.ascontiguousarray(imgs, dtype=np.float32)
#         lb_pre = np.ascontiguousarray(lb_pre, dtype=np.float32)
#         lb_post = np.ascontiguousarray(lb_post, dtype=np.float32)
#         # weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
#         return imgs, lb_pre, lb_post

#     def __len__(self):
#         return self.iters_num

#     def reset_output(self):
#         self.out_pre = np.zeros(tuple(self.raw_data_shape), dtype=np.float32)
#         self.out_post = np.zeros(tuple(self.raw_data_shape), dtype=np.float32)
#         self.weight_map = np.zeros(tuple(self.raw_data_shape), dtype=np.float32)

#     def get_weight(self, sigma=0.2, mu=0.0):
#         zz, yy, xx = np.meshgrid(np.linspace(-1, 1, self.out_size[0], dtype=np.float32),
#                                 np.linspace(-1, 1, self.out_size[1], dtype=np.float32),
#                                 np.linspace(-1, 1, self.out_size[2], dtype=np.float32), indexing='ij')
#         dd = np.sqrt(zz * zz + yy * yy + xx * xx)
#         weight = 1e-6 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2)))
#         # weight = weight[np.newaxis, ...]
#         return weight

#     def add_vol(self, pred_pre, pred_post):
#         fromz, fromy, fromx = self.pos

#         self.out_pre[fromz:fromz+self.out_size[0], \
#                         fromy:fromy+self.out_size[1], \
#                         fromx:fromx+self.out_size[2]] += pred_pre * self.weight_vol
#         self.out_post[fromz:fromz+self.out_size[0], \
#                         fromy:fromy+self.out_size[1], \
#                         fromx:fromx+self.out_size[2]] += pred_post * self.weight_vol
#         self.weight_map[fromz:fromz+self.out_size[0], \
#                         fromy:fromy+self.out_size[1], \
#                         fromx:fromx+self.out_size[2]] += self.weight_vol

#     def get_results(self):
#         self.out_pre = self.out_pre / self.weight_map
#         self.out_post = self.out_post / self.weight_map
#         if self.padding_zyx[0] == 0:
#             out_pre = self.out_pre
#             out_post = self.out_post
#         else:
#             out_pre = self.out_pre[self.padding_zyx[0]:-self.padding_zyx[0], \
#                                     self.padding_zyx[1]:-self.padding_zyx[1], \
#                                     self.padding_zyx[2]:-self.padding_zyx[2]]
#             out_post = self.out_post[self.padding_zyx[0]:-self.padding_zyx[0], \
#                                     self.padding_zyx[1]:-self.padding_zyx[1], \
#                                     self.padding_zyx[2]:-self.padding_zyx[2]]
#         return out_pre, out_post

#     def get_gt(self):
#         return self.labels_pre_origin, self.labels_post_origin

#     def get_raw(self):
#         out = self.datasets.copy()
#         if self.padding_zyx[0] != 0:
#             out = out[self.padding_zyx[0]:-self.padding_zyx[0], \
#                     self.padding_zyx[1]:-self.padding_zyx[1], \
#                     self.padding_zyx[2]:-self.padding_zyx[2]]
#         return out