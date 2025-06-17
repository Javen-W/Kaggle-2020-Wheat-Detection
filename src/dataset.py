import numpy as np
import torch
import pandas as pd
import cv2
import os
import ast
from albumentations import HorizontalFlip, VerticalFlip, ToGray, OneOf, GaussNoise, MotionBlur, MedianBlur, Blur, CLAHE, \
    RandomBrightnessContrast, Sharpen, Emboss, HueSaturationValue, BboxParams, Compose, RandomResizedCrop, Rotate
from torch.nn.functional import DType
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def compose_aug(aug, mode='train'):
    if mode == 'train':
        return Compose(aug, bbox_params=BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['labels']))
    return None

class WheatDataset(Dataset):
    def __init__(self, mode='train', data_dir='dataset/'):
        self.mode = mode
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'train' if mode in ['train', 'val'] else 'test')

        if mode in ['train', 'val']:
            # Load train.csv
            self.df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
            self.df['bbox'] = self.df['bbox'].apply(ast.literal_eval)
            self.df['image_id'] = self.df['image_id'].astype(str)

            # Extract x_min, y_min, width, height
            self.df[['x_min', 'y_min', 'w', 'h']] = pd.DataFrame(self.df.bbox.tolist(), index=self.df.index)
            self.df['x_max'] = self.df['x_min'] + self.df['w']
            self.df['y_max'] = self.df['y_min'] + self.df['h']

            # Reference images
            image_ids = list(self.df['image_id'].unique())
            train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=777)
            self.image_ids = train_ids if mode == 'train' else val_ids
        else:  # Test
            self.df = None
            self.image_ids = [os.path.splitext(f)[0] for f in os.listdir(self.image_dir) if f.endswith('.jpg')]

        # Init training transforms
        self.train_transforms = compose_aug([
            RandomResizedCrop(size=(1024, 1024), scale=(0.8, 1.0), p=0.5),
            Rotate(limit=15, p=0.5),
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
        if image is None:
            raise FileNotFoundError(f"Image {image_path} not found")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode in ['train', 'val']:
            # Get bounding boxes and labels for this image
            records = self.df[self.df['image_id'] == image_id]
            boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
            labels = torch.ones((len(boxes),), dtype=torch.int64)  # Single class (wheat)

            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx], dtype=torch.int64)}

            # Apply transforms
            if self.train_transforms:
                sample = self.train_transforms(image=image, bboxes=boxes, labels=labels)
                image = sample['image']
                boxes = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
                labels = torch.as_tensor(sample['labels'], dtype=torch.int64)
                target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx], dtype=torch.int64)}
        else:  # Test
            target = {'image_id': torch.tensor([idx], dtype=torch.int64)}

        # Convert image to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image, target