import os 
import cv2
import glob
import numpy as np
from model import unet_model

def load_model(model_path):
    model = unet_model()
    model.load_weights(model_path)
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256)) 
    image = image / 255.0 
    return image

def save_result(image, output_folder, original_image_name):
    output_path = os.path.join(output_folder, original_image_name)
    cv2.imwrite(output_path, image)

def run_inference(model_path, folder_path, output_folder):
    model = load_model(model_path)

    for image_path in glob.glob(os.path.join(folder_path, '*.jpeg')):  
        original_image_name = os.path.basename(image_path)
        image = preprocess_image(image_path)
        pred_mask = model.predict(np.expand_dims(image, axis=0))[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255 
        save_result(pred_mask, output_folder, original_image_name)

if __name__ == '__main__':
    model_path = 'Image-Segmentation/unet_road_segmentation.keras'
    test_images_folder = 'Image-Segmentation/Data/Test/images'
    output_folder = 'Image-Segmentation/Result'
    run_inference(model_path, test_images_folder, output_folder)
