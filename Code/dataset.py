import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

class RoadSegmentationDataset(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, img_size):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.image_paths = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.mask_paths = [os.path.join(mask_dir, x) for x in os.listdir(mask_dir)]

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mask_paths = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = [cv2.resize(cv2.imread(path), self.img_size) / 255.0 for path in batch_image_paths]
        masks = [cv2.resize(cv2.imread(path, 0), self.img_size) / 255.0 for path in batch_mask_paths]
        return np.array(images), np.array(masks)
