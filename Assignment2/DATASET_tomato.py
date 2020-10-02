import os
import numpy as np
import torch
from PIL import Image
from utils_label import get_label



class TomatoDataset(object):
    '''
        create a Dataset class which will be feeded 
        to a DataLoader
    '''
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'assignment_imgs'))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "assignment_imgs", self.imgs[idx])
        label = get_label(self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        
        
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(label)
        
        return img, label

    def __len__(self):
        return len(self.imgs)
