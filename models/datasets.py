import random
import os
import numpy as np
from scipy.ndimage import zoom
from itertools import chain
import torch
import scipy.io as spio
from torch.utils.data import Dataset

class Art_nosie_Dataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.mode = mode
        if mode == 'train':
            self.data_lsit = self.train_data_generator(os.path.join(data_dir, 'train'))  # patches 的集合
        else: # mode == test
            self.data_lsit = self.test_data_generator(os.path.join(data_dir, 'test'))  # patches 的集合

    def __getitem__(self, index):       
        clean_data = self.data_lsit[index % len(self.data_lsit)]       
        clean_data = torch.from_numpy(np.ascontiguousarray(clean_data)).permute(2, 0, 1).float()
        if self.mode == 'train':
            noise_data = clean_data + torch.normal(mean=0, std=random.uniform(0.05, 0.20), size=clean_data.size())
            
            if torch.rand(1).item() < 0.5:
                missing_rate = random.uniform(0.3, 0.7)  # 30%-80% 的列将随机缺失
                noise_data = self.random_missing_columns(noise_data, missing_rate)
            else:
                missing_rate = random.uniform(0.05, 0.15)  # 10%-20% 的列将连续缺失
                noise_data = self.continuous_missing_columns(noise_data, missing_rate)
        else:
            noise_data = clean_data
        
        return {'A': clean_data, 'B': noise_data}

    def __len__(self):
        return len(self.data_lsit)

    def resize_data(self, data, target_size):      
        original_size = data.shape
        zoom_factors = [target_size[i] / original_size[i] for i in range(len(original_size))]
        resize_data = zoom(data, zoom=zoom_factors, order=3)
        return resize_data

    def normalize_seismic(self, x_clean):
        old_max = np.max(np.abs(x_clean))
        x_clean = x_clean / old_max
        return x_clean

    def random_missing_columns(self, tensor, missing_rate):
        if not (0 <= missing_rate <= 1):
            raise ValueError("missing_rate 应该在 0 到 1 之间")
        num_columns = tensor.size(2)
        column_mask = torch.rand(num_columns) < missing_rate
        tensor_with_missing = tensor.clone()
        tensor_with_missing[:, :, column_mask] = 0
        return tensor_with_missing

    def continuous_missing_columns(self, tensor, missing_percentage):
        if not (0 <= missing_percentage <= 1):
            raise ValueError("missing_percentage 应该在 0 到 1 之间")
        num_columns = tensor.size(2) 
        num_missing_columns = int(num_columns * missing_percentage)    
        max_start_col = num_columns - num_missing_columns
        start_col = torch.randint(0, max_start_col + 1, (1,)).item() 
        end_col = start_col + num_missing_columns   
        tensor_with_missing = tensor.clone()
        tensor_with_missing[:, :, start_col:end_col] = 0
        return tensor_with_missing

    def gen_patches(self, img, patch_size=48, n=128, aug=True, aug_plus=False):
        '''
        :param img: input_img
        :param patch_size:
        :param n: a img generate n patches
        :param aug: if need data augmentation or not
        :return: a list of patches
        '''
        patches = list()

        ih, iw, _ = img.shape

        ip = patch_size

        for _ in range(0, n): 
            iy = random.randrange(0, ih - ip + 1)
            ix = random.randrange(0, iw - ip + 1)
            # --------------------------------
            # get patch
            # --------------------------------
            patch = img[iy:iy+ip, ix:ix+ip, :]
            patches.append(patch)

        return patches


    def train_data_generator(self, data_dir):
        data_list = list()

        filelist = os.listdir(data_dir)

        for data_name in filelist:
            path_data = os.path.join(data_dir, data_name)
            data = spio.loadmat(path_data, struct_as_record=False, squeeze_me=True)
            if os.path.normpath(data_dir) == os.path.normpath('data/train'):
                data = data['shot_data']
            data = self.resize_data(data, (256,256))
            data = self.normalize_seismic(data)
            data = np.expand_dims(data, axis=2)

            patches = self.gen_patches(data, patch_size=256, n=5)

            data_list.append(patches)

        data_list = list(chain(*data_list))

        return data_list

    
    def test_data_generator(self, data_dir):
        data_list = list()

        filelist = os.listdir(data_dir)

        for data_name in filelist:
            path_data = os.path.join(data_dir, data_name)
            data = spio.loadmat(path_data, struct_as_record=False, squeeze_me=True)
            if os.path.normpath(data_dir) == os.path.normpath('data/test'):
                data = data['shot_data']
            data = self.resize_data(data, (256,256))
            # data = self.normalize_seismic(data)
            data = np.expand_dims(data, axis=2)

            patches = self.gen_patches(data, patch_size=256, n=1)

            data_list.append(patches)

        data_list = list(chain(*data_list))

        return data_list
