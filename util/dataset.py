import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset

import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(data_root=None):    
    covid_image_path = os.path.join(data_root, 'COVID-19/images/') 
    normal_image_path = os.path.join(data_root, 'Normal/images/')
    pneumonia_image_path =  os.path.join(data_root,'Non-COVID/images/')

    print("processing datasets!")

    # Paths to masks
    covid_mask_path = os.path.join(data_root, 'COVID-19/lung masks/') 
    normal_mask_path = os.path.join(data_root, 'Normal/lung masks/') 
    pneumonia_mask_path = os.path.join(data_root, 'Non-COVID/lung masks/') 

    # All paths to images and masks
    all_image_paths = [[covid_image_path + file for file in os.listdir(covid_image_path)]
                    ,[normal_image_path + file for file in os.listdir(normal_image_path)]
                    ,[pneumonia_image_path + file for file in os.listdir(pneumonia_image_path)]
                    ]
    all_mask_paths = [[covid_mask_path + file for file in os.listdir(covid_mask_path)]
                    , [normal_mask_path + file for file in os.listdir(normal_mask_path)]
                    , [pneumonia_mask_path + file for file in os.listdir(pneumonia_mask_path)]
                    ]

    tuple_list = []
    for cls in range(len(all_image_paths)):
        for idx in range(len(all_image_paths[cls])):
            image_name = all_image_paths[cls][idx]
            label_name = all_mask_paths[cls][idx]
            assert is_image_file(image_name)
            assert is_image_file(label_name)
            cls_label = torch.zeros([3], dtype=torch.long)
            cls_label[cls] = 1
            item = (image_name, label_name, cls) # (image, label, class)
            tuple_list.append(item)  
                    
    print("{} (imagee, label, class) tuples after processing! ".format(len(tuple_list)))
    return tuple_list



class SemData(Dataset):
    def __init__(self, data_root=None, transform=None, mode='train'):
        assert mode in ['train', 'val', 'test']        
        self.mode = mode
        self.data_root = data_root

        if self.mode == 'train':
            self.data_root = os.path.join(self.data_root, 'Train')
        elif self.mode == 'val':
            self.data_root = os.path.join(self.data_root, 'Val')
        else:
            self.data_root = os.path.join(self.data_root, 'Test')

        self.data_list = make_dataset(self.data_root)
        self.transform = transform


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        image_path, label_path, cls = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = label / 255  

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))             
        
        raw_label = label.copy()
        if self.transform is not None:
            image, label = self.transform(image, label)

        if self.mode == 'train':
            return image, label, cls
        else:
            return image, label, cls, raw_label

