import os
import cv2
import sys
import numpy as np
import tensorflow as tf

def load_model(model_path):
    """
    Load the pre-trained model.
    """
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess the image: resize and scale.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

def predict(model, image):
    """
    Predict the segmentation mask of the image.
    """
    image = np.expand_dims(image, axis=0)
    pred_mask = model.predict(image)
    return pred_mask[0]

def save_result(image, result_path):
    """
    Save the segmentation result.
    """
    cv2.imwrite(result_path, image * 255)

def main(image_path, model_path, result_path):
    """
    Main function to handle the inference pipeline.
    """
    model = load_model(model_path)
    image = preprocess_image(image_path)
    segmented_image = predict(model, image)
    save_result(segmented_image, result_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python inference.py <image_path> <model_path> <result_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2]
    result_path = sys.argv[3]

    main(image_path, model_path, result_path)
