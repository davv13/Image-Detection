import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image

def load_model(model_path):
    """
    Load a pre-trained segmentation model from the specified path.

    Parameters:
    - model_path (str): The path to the pre-trained segmentation model saved in a format compatible with Keras.

    Returns:
    - keras.Model: The loaded segmentation model.
    """
    model = keras.models.load_model(model_path)
    return model

def preprocess_input(image_path):
    """
    Preprocesses an input image for segmentation.

    This function loads an image from the specified path, resizes it to the target size,
    converts it to a NumPy array, expands the dimensions to create a batch, and normalizes
    pixel values to the range [0, 1].

    Parameters:
    - image_path (str): The path to the input image file.

    Returns:
    - numpy.ndarray: The preprocessed image as a NumPy array.
    """
    img = image.load_img(image_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array


def postprocess_output(output_tensor, threshold=0.5):
    """
    Postprocesses the output tensor from a segmentation model.

    This function converts the probability map in the output tensor to a binary mask using
    a specified threshold. Pixels with values above the threshold are set to 0 (black), and
    pixels below the threshold are set to 1 (white).

    Parameters:
    - output_tensor (numpy.ndarray): The output tensor from a segmentation model.
    - threshold (float, optional): The probability threshold for binary segmentation.
      Default is 0.5.

    Returns:
    - numpy.ndarray: The binary mask representing the segmented areas.
    """
    binary_mask = (output_tensor > threshold).astype(np.uint8)
    binary_mask = 1 - binary_mask 

    return binary_mask


def save_result(output_tensor, output_folder, image_name):
    """
    Saves the binary segmentation result to an image file.

    This function takes a binary mask (output tensor) representing the segmented areas,
    converts it to a grayscale PIL Image, and saves it to the specified output folder with
    the given image name.

    Parameters:
    - output_tensor (numpy.ndarray): Binary mask representing the segmented areas.
    - output_folder (str): The folder where the result image will be saved.
    - image_name (str): The name of the output image file.
    """
    result_image = Image.fromarray(output_tensor * 255, mode='L') 

    result_path = os.path.join(output_folder, image_name)
    result_image.save(result_path)

def inference(image_path, model, output_folder):
    """
    Perform image segmentation on a single input image using a pre-trained model.

    This function takes an input image, preprocesses it, performs inference using a pre-trained
    segmentation model, postprocesses the model output, and saves the segmented result to a file.

    Parameters:
    - image_path (str): Path to the input image for segmentation.
    - model: The pre-trained segmentation model.
    - output_folder (str): The folder where the segmentation result will be saved.
    """
    input_tensor = preprocess_input(image_path)

    output_tensor = model.predict(input_tensor)

    output_tensor = postprocess_output(output_tensor)

    save_result(output_tensor, output_folder, os.path.basename(image_path))


if __name__ == "__main__":
    """
    Example:
    >>> python inference.py --image_path path/to/your/image.jpg --checkpoint_path path/to/your/checkpoint.pth --output_folder Result
    """
    parser = argparse.ArgumentParser(description="Inference script for image segmentation")
    parser.add_argument("--image_path", type=str, help="Path to the input image for inference")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained segmentation model")
    parser.add_argument("--output_folder", type=str, default="Result", help="Folder to save the inference result")

    args = parser.parse_args()

    model = load_model(args.model_path)

    inference(args.image_path, model, args.output_folder)