import os
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SegmentationDataset(tf.keras.utils.Sequence):
    """
    Custom dataset class for image segmentation.

    This class inherits from tf.keras.utils.Sequence and is designed to be used
    with the tf.keras model training process.

    Parameters:
    - data_folder (str): Path to the dataset folder containing subfolders for images and masks.
    - batch_size (int): Batch size for training.
    - image_size (tuple): Target size for resizing input images.
    """
    def __init__(self, data_folder, batch_size=32, image_size=(256, 256)):
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.image_size = image_size

        # List of image and mask file paths
        self.image_paths = glob.glob(os.path.join(data_folder, "images", "*.jpg"))
        self.mask_paths = glob.glob(os.path.join(data_folder, "masks", "*.png"))

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
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        batch_image_paths = self.image_paths[start_idx:end_idx]
        batch_mask_paths = self.mask_paths[start_idx:end_idx]

        batch_images = [tf.keras.preprocessing.image.load_img(path, target_size=self.image_size) for path in batch_image_paths]
        batch_masks = [tf.keras.preprocessing.image.load_img(path, target_size=self.image_size, color_mode="grayscale") for path in batch_mask_paths]

        # Convert images and masks to arrays
        batch_images = np.array([tf.keras.preprocessing.image.img_to_array(img) for img in batch_images])
        batch_masks = np.array([tf.keras.preprocessing.image.img_to_array(mask) for mask in batch_masks])

        # Augment images and masks (if needed)
        # ...

        return batch_images, batch_masks
