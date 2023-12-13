import os
import cv2
import glob
import numpy as np
import tensorflow as tf

class RoadSegmentationDataset(tf.data.Dataset):
    def __new__(cls, data_dir, image_size=(256, 256), mode='training'):
        cls.image_size = image_size
        cls.mode = mode
        return cls._create_dataset(data_dir)

    @classmethod
    def _create_dataset(cls, data_dir):
        image_paths, label_paths = cls._load_data_paths(data_dir, cls.mode)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        dataset = dataset.map(cls._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    @classmethod
    def _load_data_paths(cls, data_dir, mode):
        image_dir = os.path.join(data_dir, mode, mode)
        label_dir = os.path.join(data_dir, mode, f"{mode}_labels")

        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tiff")))
        label_paths = sorted(glob.glob(os.path.join(label_dir, "*.tiff")))

        return image_paths, label_paths

    @classmethod
    def _process_path(cls, image_path, label_path):
        image = cls._load_and_preprocess_image(image_path)
        label = cls._load_and_preprocess_label(label_path)
        return image, label

    @classmethod
    def _load_and_preprocess_image(cls, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, cls.image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    @classmethod
    def _load_and_preprocess_label(cls, path):
        label = tf.io.read_file(path)
        label = tf.image.decode_image(label, channels=1)
        label = tf.image.resize(label, cls.image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return label

def load_data(data_dir, batch_size=32, image_size=(256, 256), mode='training'):
    dataset = RoadSegmentationDataset(data_dir, image_size=image_size, mode=mode)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
