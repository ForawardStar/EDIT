import glob
import random
import os

from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.domains_num = 3

        self.photo_files = sorted(glob.glob(os.path.join(root, 'monet2photo/%sB' % mode) + '/*.*'))
        self.monet_files = sorted(glob.glob(os.path.join(root, 'monet2photo/%sA' % mode) + '/*.*'))
        #self.ukiyoe_files = sorted(glob.glob(os.path.join(root, 'ukiyoe2photo/%sA' % mode) + '/*.*'))
        self.sheos_files_A = sorted(glob.glob(os.path.join(root,'real_sheos') + '/*.*'))
        self.sheos_files_B = sorted(glob.glob(os.path.join(root, 'edge_sheos') + '/*.*'))
        self.building_files_A = sorted(glob.glob(os.path.join(root, 'building_semantic_image') + '/*.*'))
        self.building_files_B = sorted(glob.glob(os.path.join(root, 'building_semantic_map') + '/*.*'))
        #self.bag_files_A = sorted(glob.glob(os.path.join(root, 'EdgesndHandbags/bags') + '/*.*'))
        #self.bag_files_B = sorted(glob.glob(os.path.join(root, 'EdgesndHandbags/edges') + '/*.*'))

    def __getitem__(self, index):
        label = random.randint(1, self.domains_num)

        if label == 1:
            item_A = self.transform(Image.open(self.sheos_files_A[index % len(self.sheos_files_A)]).convert("RGB"))

            item_B = self.transform(Image.open(self.sheos_files_B[random.randint(0, len(self.sheos_files_B) - 1)]).convert("RGB"))
        elif label == 2:
            item_A = self.transform(Image.open(self.building_files_A[index % len(self.building_files_A)]).convert("RGB"))

            item_B = self.transform(Image.open(self.building_files_B[random.randint(0, len(self.building_files_B) - 1)]).convert("RGB"))
        elif label == 3:
            item_A = self.transform(Image.open(self.monet_files[index % len(self.monet_files)]).convert("RGB"))

            item_B = self.transform(Image.open(self.photo_files[random.randint(0, len(self.photo_files) - 1)]).convert("RGB"))
        #elif label == 4:
        #    item_A = self.transform(Image.open(self.ukiyoe_files[index % len(self.ukiyoe_files)]).convert("RGB"))

        #    item_B = self.transform(Image.open(self.photo_files[random.randint(0, len(self.photo_files) - 1)]).convert("RGB"))

        one_hot_label_AB = np.zeros((self.domains_num + 1, item_A.shape[1], item_A.shape[2]))
        one_hot_label_BA = np.zeros((self.domains_num + 1, item_A.shape[1], item_A.shape[2]))
        one_hot_label_AB[label-1] = 1
        one_hot_label_BA[label-1] = 1
        one_hot_label_AB[self.domains_num] = 0
        one_hot_label_BA[self.domains_num] = 1

        return {'A': item_A, 'B': item_B, 'label_AB':one_hot_label_AB, 'label_BA':one_hot_label_BA, 'label_name':label}

    def __len__(self):
        return max(len(self.sheos_files_A), len(self.sheos_files_B))
