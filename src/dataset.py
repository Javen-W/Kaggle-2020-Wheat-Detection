import numpy as np
import torch
import pandas as pd
import cv2
import os
import ast
from albumentations import HorizontalFlip, VerticalFlip, ToGray, OneOf, GaussNoise, MotionBlur, MedianBlur, Blur, CLAHE, \
    RandomBrightnessContrast, Sharpen, Emboss, HueSaturationValue, BboxParams, Compose
from torch.utils.data import Dataset

def compose_aug(aug):
    return Compose(aug, bbox_params=BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['labels']))

class WheatDataset(Dataset):
    def __init__(self, mode='train'):
        # Load train.csv
        self.df = pd.read_csv('dataset/train.csv')
        self.df['bbox'] = self.df['bbox'].apply(ast.literal_eval)

        # Extract x_min, y_min, width, height
        self.df[['x_min', 'y_min', 'w', 'h']] = pd.DataFrame(self.df.bbox.tolist(), index=self.df.index)
        self.df['x_max'] = self.df['x_min'] + self.df['w']
        self.df['y_max'] = self.df['y_min'] + self.df['h']

        # Reference images
        self.image_ids = list(self.df['image_id'].unique())
        self.image_dir = 'dataset/train' if mode == 'train' else 'dataset/test'

        # Init training transforms
        self.train_transforms = compose_aug([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ToGray(p=0.01),
            GaussNoise(p=0.2),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            OneOf([
                CLAHE(),
                Sharpen(),
                Emboss(),
                RandomBrightnessContrast(),
            ], p=0.25),
            HueSaturationValue(p=0.25)
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get bounding boxes and labels for this image
        records = self.df[self.df['image_id'] == image_id]
        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # Single class (wheat)

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}

        # Apply transforms
        if self.train_transforms:
            annotated = {'image': image, 'labels': labels, 'boxes': boxes}
            image = self.train_transforms(**annotated)['image']

        # Convert image to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # Use permute instead of moveaxis

        return image, target