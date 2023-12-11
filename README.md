# Image Segmentation Project

This project involves the creation of a segmentation model for road detection in satellite images using TensorFlow and Keras.

## Project Structure

The project is organized into the following files and folders:

- **Code Folder:**
  - `inference.py`: Contains the model inference code for processing a single image.
  - `dataset.py`: Includes code related to the dataset, such as the `SegmentationDataset` class, preprocessing, augmentation, and post-processing routines.
  - `model.py`: Contains code related to the model, including the network architecture, layers, and loss functions.
  - `train.py`: Contains the training code. It accepts a dataset path and hyperparameters as inputs and produces and saves at least one checkpoint as output.

- **Data Folder:**
  - `training`: Contains data used for training.
  - `validation`: Contains data used for validation.
  - `test`: For holding data used for testing.

- **Result Folder:**
  - Contains the results of testing.

## Getting Started

1. **Install Dependencies:**
   ```bash
    pip install -r requirements.txt
2. **Training the Model:**
   ```bash
    python train.py --dataset_path Data --output_model_path model.h5 --epochs 10 --batch_size 32
3. **Model Inference:**
   ```bash
   python inference.py --image_path Data/test/images/test_image.jpg --model_path model.h5 --output_folder Result
