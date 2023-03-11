import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
import torchvision.transforms.functional as tf
from typing import Dict
from torch import Tensor
import numpy as np
import glob
import cv2
import json
from PIL import Image
import random
import torch

class CropCityscapesArtefacts:
    """Crop Cityscapes images to remove artefacts"""

    def __init__(self):
        self.top = 64
        self.left = 128
        self.right = 128
        self.bottom = 256

    def __call__(self, image):
        """Crops a PIL image.
        Args:
            image (PIL.Image): Cityscapes image (or disparity map)
        Returns:
            PIL.Image: Cropped PIL Image
        """
        w, h = image.size
        assert w == 2048 and h == 1024, f'Expected (2048, 1024) image but got ({w}, {h}). Maybe the ordering of transforms is wrong?'
        #w, h = 1792, 704
        return image.crop((self.left, self.top, w-self.right, h-self.bottom))

class MinimalCrop:
    """
    Performs the minimal crop such that height and width are both divisible by min_div.
    """
    
    def __init__(self, min_div=16):
        self.min_div = min_div
        
    def __call__(self, image):
        w, h = image.size
        
        h_new = h - (h % self.min_div)
        w_new = w - (w % self.min_div)
        
        if h_new == 0 and w_new == 0:
            return image
        else:    
            h_diff = h-h_new
            w_diff = w-w_new

            top = int(h_diff/2)
            bottom = h_diff-top
            left = int(w_diff/2)
            right = w_diff-left

            return image.crop((left, top, w-right, h-bottom))

class StereoImageDataset(Dataset):
    """Dataset class for image compression datasets."""
    #/home/xzhangga/datasets/Instereo2K/train/
    def __init__(self, ds_type='train', ds_name='cityscapes', root='/home/xzhangga/datasets/Cityscapes/', crop_size=(256, 256), resize=False, **kwargs):
        """
        Args:
            name (str): name of dataset, template: ds_name#ds_type. No '#' in ds_name or ds_type allowed. ds_type in (train, eval, test).
            path (str): if given the dataset is loaded from path instead of by name.
            transforms (Transform): transforms to apply to image
            debug (bool, optional): If set to true, limits the list of files to 10. Defaults to False.
        """
        super().__init__()
        
        self.path = Path(f"{root}")
        self.ds_name = ds_name
        self.ds_type = ds_type
        if ds_type=="train":
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
                transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
        else: 
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.left_image_list, self.right_image_list = self.get_files()


        if ds_name == 'cityscapes':
            self.crop = CropCityscapesArtefacts()
        else:
            if ds_type == "test":
                self.crop = MinimalCrop(min_div=64)
            else:
                self.crop = None
        #self.index_count = 0

        print(f'Loaded dataset {ds_name} from {self.path}. Found {len(self.left_image_list)} files.')

    def __len__(self):
        return len(self.left_image_list)

    def __getitem__(self, index):
        #self.index_count += 1
        image_list = [Image.open(self.left_image_list[index]).convert('RGB'), Image.open(self.right_image_list[index]).convert('RGB')]
        if self.crop is not None:
            image_list = [self.crop(image) for image in image_list]
        frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
        frames = torch.chunk(self.transform(frames), 2)
        if random.random() < 0.5:
            frames = frames[::-1]
        return frames

    def get_files(self):
        if self.ds_name == 'cityscapes':
            left_image_list, right_image_list, disparity_list = [], [], []
            for left_image_path in self.path.glob(f'leftImg8bit/{self.ds_type}/*/*.png'):
                left_image_list.append(str(left_image_path))
                right_image_list.append(str(left_image_path).replace("leftImg8bit", 'rightImg8bit'))
                disparity_list.append(str(left_image_path).replace("leftImg8bit", 'disparity'))

        elif self.ds_name == 'instereo2k':
            path = self.path / self.ds_type
            if self.ds_type == "test":
                folders = [f for f in path.iterdir() if f.is_dir()]
            else:
                folders = [f for f in path.glob('*/*') if f.is_dir()]
            left_image_list = [f / 'left.png' for f in folders]
            right_image_list = [f / 'right.png' for f in folders]

        elif self.ds_name == 'kitti':
            left_image_list, right_image_list = [], []
            ds_type = self.ds_type + "ing"
            for left_image_path in self.path.glob(f'stereo2012/{ds_type}/colored_0/*.png'):
                left_image_list.append(str(left_image_path))
                right_image_list.append(str(left_image_path).replace("colored_0", 'colored_1'))

            for left_image_path in self.path.glob(f'stereo2015/{ds_type}/image_2/*.png'):
                left_image_list.append(str(left_image_path))
                right_image_list.append(str(left_image_path).replace("image_2", 'image_3'))

        elif self.ds_name == 'wildtrack':
            C1_image_list, C4_image_list = [], []
            for image_path in self.path.glob(f'images/C1/*.png'):
                if self.ds_type == "train" and int(image_path.stem) <= 2000:
                    C1_image_list.append(str(image_path))
                    C4_image_list.append(str(image_path).replace("C1", 'C4'))
                elif self.ds_type == "test" and int(image_path.stem) > 2000:
                    C1_image_list.append(str(image_path))
                    C4_image_list.append(str(image_path).replace("C1", 'C4'))
            left_image_list, right_image_list = C1_image_list, C4_image_list
        else:
            raise NotImplementedError

        return left_image_list, right_image_list

class MultiCameraImageDataset(Dataset):
    def __init__(self, ds_type='train', ds_name='wildtrack', root='/home/xzhangga/datasets/WildTrack/', crop_size=(256, 256), num_camera=7, **kwargs):
        super().__init__()
        
        self.path = Path(f"{root}")
        self.ds_name = ds_name
        self.ds_type = ds_type
        if ds_type=="train":
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
                transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
        else: 
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.num_camera = num_camera
        self.image_lists = self.get_files()
        if ds_type == "test":
            self.crop = MinimalCrop(min_div=64)
        else:
            self.crop = None

        print(f'Loaded dataset {ds_name} from {self.path}. Found {len(self.image_lists[0])} files.')

    def __len__(self):
        return len(self.image_lists[0])

    def __getitem__(self, index):
        image_list = [Image.open(self.image_lists[i][index]).convert('RGB') for i in range(self.num_camera)]
        if self.crop is not None:
            image_list = [self.crop(image) for image in image_list]
        frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
        frames = torch.chunk(self.transform(frames), self.num_camera)
        #if random.random() < 0.5:
        #    frames = frames[::-1]
        return frames

    def set_stage(self, stage):
        if stage == 0:
            print('Using (32, 32) crops')
            self.crop = transforms.RandomCrop((32, 32))
        elif stage == 1:
            print('Using (28, 28) crops')
            self.crop = transforms.RandomCrop((28, 28))

    def get_files(self):
        if self.ds_name == 'wildtrack':
            image_lists = [[] for i in range(self.num_camera)]
            for image_path in self.path.glob(f'images/C1/*.png'):
                if self.ds_type == "train" and int(image_path.stem) <= 2000:
                    image_lists[0].append(str(image_path))
                    for idx in range(1, self.num_camera):
                        image_lists[idx].append(str(image_path).replace("C1", 'C'+str(idx+1))) 
                elif self.ds_type == "test" and int(image_path.stem) > 2000:
                    image_lists[0].append(str(image_path))
                    for idx in range(1, self.num_camera):
                        image_lists[idx].append(str(image_path).replace("C1", 'C'+str(idx+1)))
        else:
            raise NotImplementedError

        return image_lists

class AdaptiveMultiCameraImageDataset(Dataset):
    def __init__(self, ds_type='train', ds_name='wildtrack', root='/home/xzhangga/datasets/WildTrack/', crop_size=(256, 256), **kwargs):
        super().__init__()
        
        self.path = Path(f"{root}")
        self.ds_name = ds_name
        self.ds_type = ds_type
        if ds_type=="train":
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
                transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
        else: 
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_lists = self.get_files()
        self.set_num_camera()
        if ds_type == "test":
            self.crop = MinimalCrop(min_div=64)
            self.num_camera = 7
        else:
            self.crop = None

        print(f'Loaded dataset {ds_name} from {self.path}. Found {len(self.image_lists[0])} files.')

    def __len__(self):
        return len(self.image_lists[0])

    def __getitem__(self, index):
        if self.ds_type == "train":
            image_list = [Image.open(self.image_lists[i][index]).convert('RGB') for i in self.images_index]
        else:
            image_list = [Image.open(self.image_lists[i][index]).convert('RGB') for i in range(self.num_camera)]
        if self.crop is not None:
            image_list = [self.crop(image) for image in image_list]
        frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
        frames = torch.chunk(self.transform(frames), self.num_camera)
        #if random.random() < 0.5:
        #    frames = frames[::-1]
        return frames

    def set_num_camera(self):
        self.num_camera = random.randint(2, 7)
        self.images_index = random.sample(range(7), self.num_camera)
        #print("num_camera:",self.num_camera)

    def get_files(self, num_camera=7):
        if self.ds_name == 'wildtrack':
            image_lists = [[] for i in range(num_camera)]
            for image_path in self.path.glob(f'images/C1/*.png'):
                if self.ds_type == "train" and int(image_path.stem) <= 2000:
                    image_lists[0].append(str(image_path))
                    for idx in range(1, num_camera):
                        image_lists[idx].append(str(image_path).replace("C1", 'C'+str(idx+1))) 
                elif self.ds_type == "test" and int(image_path.stem) > 2000:
                    image_lists[0].append(str(image_path))
                    for idx in range(1, num_camera):
                        image_lists[idx].append(str(image_path).replace("C1", 'C'+str(idx+1)))
        else:
            raise NotImplementedError

        return image_lists


def save_checkpoint(state, is_best=False, log_dir=None, filename="ckpt.pth.tar"):
    save_file = os.path.join(log_dir, filename)
    print("save model in:", save_file)
    torch.save(state, save_file)
    if is_best:
        torch.save(state, os.path.join(log_dir, filename.replace(".pth.tar", ".best.pth.tar")))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_output_folder(parent_dir, env_name, output_current_folder=False):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    if not output_current_folder: 
        experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir, experiment_id


