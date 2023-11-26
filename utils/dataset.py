#https://github.com/bowang-lab/MedSAM
import os
import os
import pandas as pd
import numpy as np
import cv2
from typing import Any, Tuple,List
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import glob
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

class SamDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_col: str,
        mask_col: str,
        image_dir: Any = None,
        mask_dir: str = None,
        image_size: Tuple = (256,256),
        dataset_type: str = None,
        process_type: str = None,

        **kwargs,
    ):
        """
        PyTorch dataset class for loading image,mask and bbox pairs from a dataframe.
        The dataframe will need to have atleast two columns for the image and mask file names. The columns can either have the full or relative
        path of the images or just the file names.
        If only file names are given in the columns, the `image_dir` and `mask_dir` arguments should be specified.

        Args:
            df (pd.DataFrame): the pandas dataframe object
            image_col (str): the name of the column on the dataframe that holds the image file names.
            mask_col (str): the name of the column on the dataframe that holds the mask file names.
            image_dir (Any, optional): Path to the input image directory. Defaults to None.
            mask_dir (str, optional): Path to the mask images directory. Defaults to None.
            image_size (Tuple, optional): image size. Defaults to (1024, 1024).
        """
        self.df = df
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_col = image_col
        self.mask_col = mask_col
        self.image_size = image_size
        self.dataset_type = dataset_type
        self.bbox_offset_min = kwargs['bbox_offset_min']
        self.bbox_offset_max = kwargs['bbox_offset_max']
        self.process_type=process_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # read dataframe row
        row = self.df.iloc[idx]
        # If the `image_dir` attribute is set, the path will be relative to that directory.
        # Otherwise, the path will be the value of the `row[self.image_col]` attribute.
        image_file = (
            os.path.join(self.image_dir, row[self.image_col])
            if self.image_dir
            else row[self.image_col]
        )
        mask_file = (
            os.path.join(self.mask_dir, row[self.mask_col])
            if self.mask_dir
            else row[self.mask_col]
        )

        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Couldn't find image {image_file}")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Couldn't find image {mask_file}")

        

        if self.dataset_type == 'camo':

            image_data = Image.open(image_file)
            mask_data = Image.open(mask_file)

            return self._preprocess_camo(image_data, mask_data)
        else:

            # read image and mask files
            image_data = cv2.imread(image_file)
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

            # read mask as gray scale
            # mask_data = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask_data = cv2.imread(mask_file)
            mask_data = cv2.cvtColor(mask_data, cv2.COLOR_BGR2GRAY)
            if self.process_type == 'sam':
                return self._preprocess_sam(image_data, mask_data)

            return self._preprocess(image_data, mask_data)
    
    def _preprocess_camo(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.image_size = (1024,1024)
        self.img_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        self.mask_transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
        
        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        bbox = self._get_bbox(mask)

        return image, mask, bbox
    def _preprocess_sam(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mask.max()<127:
            mask = mask*255
            
        else:
            mask = cv2.threshold(mask, 127.0, 255.0, cv2.THRESH_BINARY)[1]

        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        image = (image - pixel_mean) / pixel_std
       
        # convert to tensor
        image = TF.to_tensor(image).float()

        # convert to tensor
      
        mask = TF.to_tensor(mask)
        
        # resize
        image = TF.resize(image, self.image_size, antialias=True)
        mask = TF.resize(mask, self.image_size, antialias=True)

        bbox = self._get_bbox(mask)

        return image, mask, bbox
    

    def _preprocess(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Threshold mask to binary
        # mask = cv2.threshold(mask, 127.0, 255.0, cv2.THRESH_BINARY)[1]
        if mask.max()<127:
            mask = mask*255
            
        else:
            mask = cv2.threshold(mask, 127.0, 255.0, cv2.THRESH_BINARY)[1]
        
        # convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        # min-max normalize and scale
        image = (image - image.min()) / (image.max() - image.min()) * 255.0
        # resize
        image = TF.resize(image, self.image_size, antialias=True)
        mask = TF.resize(mask, self.image_size, antialias=True)

        bbox = self._get_bbox(mask)

        return image, mask, bbox

    def _get_bbox(self, mask: torch.Tensor):
        _, y_indices, x_indices = torch.where(mask > 0)

        x_min, y_min = (x_indices.min(), y_indices.min())
        x_max, y_max = (x_indices.max(), y_indices.max())

        # add perturbation to bounding box coordinates
        H, W = mask.shape[1:]
        # add perfurbation to the bbox
        assert H == W, f"{W} and {H} are not equal size!!"
        x_min = max(0, x_min - np.random.randint(self.bbox_offset_min, self.bbox_offset_max))
        x_max = min(W, x_max + np.random.randint(self.bbox_offset_min, self.bbox_offset_max))
        y_min = max(0, y_min - np.random.randint(self.bbox_offset_min, self.bbox_offset_max))
        y_max = min(H, y_max + np.random.randint(self.bbox_offset_min, self.bbox_offset_max))

        return np.array([x_min, y_min, x_max, y_max])


def _create_csv(mode,path,img_ext,lbl_ext,do_split=False):
        if mode is not None:
            masks = glob.glob(f'{path}/{mode}/masks/*.{lbl_ext}')
            images=glob.glob(f'{path}/{mode}/images/*.{img_ext}')
        else:
            
            masks = glob.glob(f'{path}/masks/*.{lbl_ext}')
            images=glob.glob(f'{path}/images/*.{img_ext}')
        
        images = sorted(images)
        masks = sorted(masks)
        # print(images[0])
        print(len(images),len(masks))
       
        
        df = pd.DataFrame(
            {
                'image_path':images,
                'mask_path':masks
            }
            )
        if do_split:
            train_df, val_df = train_test_split(df, train_size=0.8, random_state=2023)
            return train_df,val_df
            
        return df





        
    
        
        
                 
        
