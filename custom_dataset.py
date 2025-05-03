

# Imports
import torch
import os
import pandas as pd
import numpy as np
# import math
from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches



class RealFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, current_dir, num_images, curr_dim, transform=None):
        self.real_dir = real_dir                 #### directory of topside images
        self.fake_dir = fake_dir                 #### directory of bottom images
        self.current_dir = current_dir           #### directory of current information
        self.transform = transform
        self.num_images = num_images             #### number of bottom images
        self.curr_dim = curr_dim                 #### dimension of current information
        

    def __len__(self):
        lst = os.listdir(self.real_dir)          #### your directory path
        
        
        number_files = len(lst) - (self.num_images + self.curr_dim - 1)
        return number_files

    def __getitem__(self, index):
        df = pd.read_csv(os.path.join(self.current_dir, 'current_new.csv'), header=None, names=['current'])
        
        
        for id_h in range(self.num_images):
            current_data = []
            
            for id_x in range(self.curr_dim):
                
                df_current = df['current'][index+id_x+id_h]
                df_current = df_current.astype(np.float32)
                current_data.append(df_current)
            current_data = torch.Tensor(current_data)
            current_data = current_data.unsqueeze(0)
            if id_h == 0:
                current_new = current_data
            else:
                current_new = torch.cat((current_new, current_data), dim=0)
        
        
        img_path = os.path.join(self.real_dir, "{}.bmp".format(index+(self.num_images+self.curr_dim)))
        image = io.imread(img_path, as_gray=True)
        if self.transform:
            image = self.transform(image)
        
        # self.fake_paths = [os.path.join(self.fake_dir, "{}.bmp".format(index+i)) for i in range(1, 61)]
        
        self.fake_paths = [os.path.join(self.fake_dir, "{}.bmp".format(index+i)) for i in range(self.curr_dim, self.num_images + self.curr_dim)]

        fake_data = []
        for fake_path in self.fake_paths:
            fake = io.imread(fake_path, as_gray=True)
            if self.transform:
                fake = self.transform(fake)
            fake_data.append(fake)

        fake_data = torch.stack(fake_data, dim=0)

        return image, fake_data.squeeze(), current_new




