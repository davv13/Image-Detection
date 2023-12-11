import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SegmentationDataset(tf.keras.utils.Sequence):
    """
    Custom dataset class for image segmentation.

    ...

    Parameters:
    - data_folder (str): Path to the dataset folder containing 'Training', 'Validation', and 'Test' subfolders.
    - batch_size (int): Batch size for training.
    - image_size (tuple): Target size for resizing input images.
    """
    def __init__(self, data_folder, batch_size=32, image_size=(256, 256)):
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.image_size = image_size

        # List of image and mask file paths for training
        self.train_image_paths = glob.glob(os.path.join(data_folder, 'Training', 'train', '*.tif'))
        self.train_mask_paths = glob.glob(os.path.join(data_folder, 'Training', 'train_labels', '*.tif'))

        # List of image and mask file paths for validation
        self.val_image_paths = glob.glob(os.path.join(data_folder, 'Validation', 'val', '*.tif'))
        self.val_mask_paths = glob.glob(os.path.join(data_folder, 'Validation', 'val_labels', '*.tif'))

        # Image data generator for augmentation (customize as needed)
        self.image_data_generator = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def __len__(self):
        # Use either the length of the training or validation set based on the context
        return len(self.train_image_paths) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        batch_image_paths = self.train_image_paths[start_idx:end_idx]
        batch_mask_paths = self.train_mask_paths[start_idx:end_idx]

        batch_images = [tf.keras.preprocessing.image.load_img(path, target_size=self.image_size) for path in batch_image_paths]
        batch_masks = [tf.keras.preprocessing.image.load_img(path, target_size=self.image_size, color_mode="grayscale") for path in batch_mask_paths]

        # Convert images and masks to arrays
        batch_images = np.array([tf.keras.preprocessing.image.img_to_array(img) for img in batch_images])
        batch_masks = np.array([tf.keras.preprocessing.image.img_to_array(mask) for mask in batch_masks])

        return batch_images, batch_masks
