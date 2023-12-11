import os
import argparse
import tensorflow as tf
from model import create_unet_model
from dataset import SegmentationDataset

def train(dataset_path, output_model_path, epochs=10, batch_size=32):
    """
    Train a segmentation model.

    Parameters:
    - dataset_path (str): Path to the dataset folder containing 'training', 'validation', and 'test' subfolders.
    - output_model_path (str): Path to save the trained segmentation model.
    - epochs (int): Number of training epochs. Default is 10.
    - batch_size (int): Batch size for training. Default is 32.
   
    Example:
    >>> python train.py --dataset_path path/to/your/dataset --output_model_path path/to/save/model.h5 --epochs 10 --batch_size 32
    """
    input_shape = (256, 256, 3)
    num_classes = 1

    model = create_unet_model(input_shape, num_classes)

    train_dataset = SegmentationDataset(os.path.join(dataset_path, 'training'))
    val_dataset = SegmentationDataset(os.path.join(dataset_path, 'validation'))

    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, batch_size=batch_size)

    model.save(output_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset folder")
    parser.add_argument("--output_model_path", type=str, help="Path to save the trained segmentation model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    args = parser.parse_args()

    train(args.dataset_path, args.output_model_path, args.epochs, args.batch_size)
