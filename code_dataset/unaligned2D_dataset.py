import random

from code_dataset.base_dataset import BaseDataset
from code_dataset.image_folder import make_dataset
from code_util.data.read_save import read_image
from code_dataset.image_folder import get_image_directory_plural, make_datset_plural
from code_util.data.prepost_process import Preprocess
from code_util.data.read_save import read_dummy
import numpy as np

class Unaligned2DDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, config):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, config)

        self.config = config
        self.dim = config["dataset"]["dim"]
        self.transform = Preprocess(self.config)

        self.make_dataset = make_dataset
        self.read_image = read_image

        self.dir_A, self.dir_B, self.dir_mask = get_image_directory_plural(self.config)  # get the image directory
        self.A_paths, self.B_paths, self.Mask_paths = make_datset_plural([self.dir_A, self.dir_B, self.dir_mask],self.make_dataset, self.config)

        # 对B进行shuffle
        # random.shuffle(self.B_paths)

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.Mask_size = len(self.Mask_paths)  # get the size of dataset Mask

        assert self.A_size == self.Mask_size, "The number of images in A and Mask directories must be equal. Please check your dataset."

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A,B,Mask = self._get_full_image(index)
        A,B,Mask = self._apply_transform(A,B,Mask)

        return {'A': A, 'B': B, 'Mask': Mask}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
    
    def _get_full_image(self, index):
        """compared to the get_patch function, this function returns the full image."""
        A_path = self.A_paths[index % self.A_size]
        Mask_path = self.Mask_paths[index % self.A_size] # 默认Mask和A是一一对应的
        # 随机选择另一个B
        random_index = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[random_index]
        
        A = self.read_image(A_path) 
        B = self.read_image(B_path) if B_path is not None else read_dummy()
        Mask = self.read_image(Mask_path) if Mask_path is not None else read_dummy()

        return A, B, Mask
    
    def _apply_transform(self,A,B,Mask):
        A_path = A['params']['path']
        B_path = B['params']['path']
        Mask_path = Mask['params']['path']
        self.transform.init_pipeline()
        A_transform = self.transform('A', A_path)
        if A_path is not None:
            A['data'] = A_transform(A['data'])
        B_transform = self.transform('B', B_path)
        if B_path is not None:
            B['data'] = B_transform(B['data'])
        Mask_transform = self.transform('Mask', Mask_path)
        if Mask_path is not None:
            Mask['data'] = Mask_transform(Mask['data'])

        return A, B, Mask
