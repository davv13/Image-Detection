# Road Segmentation Using U-Net and TensorFlow

This project implements a U-Net model for road segmentation in images, utilizing TensorFlow. The code is structured into several key scripts for training the model, performing inference, handling the dataset, and defining the model architecture.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.x
- TensorFlow 2.x
- OpenCV-Python
- NumPy

## Project Structure

The project is organized into several directories and key files:

- **Code/**
  - `inference.py`: Script for model inference on single images.
  - `dataset.py`: Handles dataset loading and preprocessing.
  - `model.py`: Defines the U-Net model architecture.
  - `train.py`: Contains the training routine.
- **Data/**
  - Sub-folders for training, validation, and testing datasets.
- **Results/**
  - Directory for storing output images from inference.

## Setup and Running Instructions

### Training the Model

1. **Prepare the Dataset**
   Place your dataset in the 'Data/' folder with the following structure:
   Data/
    ├── Train/\n
    ├── Validation/\n
    └── Test/\n

2. **Configure the Training Script**
Open `train.py` and set your dataset path and other desired hyperparameters.

3. **Run the Training**
Execute the training script:
```python
  python train.py
```

### Running Inference

1. **Prepare the Model**
Ensure your trained model is saved in `.keras` format.

2. **Configure Inference Script**
In `inference.py`, set the paths for the model, input image, and output destination.

3. **Execute Inference**
Run the script to process an image:

```python
  python inference.py
```

## Additional Notes

- Ensure all script paths are correctly set according to your directory structure.
- Modify `dataset.py` and `model.py` as needed to suit your specific project requirements.
