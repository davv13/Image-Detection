import os
import datetime
import tensorflow as tf
from model import unet_model
from dataset import RoadSegmentationDataset
from tensorflow.keras.callbacks import TensorBoard

def train(dataset_path, epochs, batch_size, learning_rate):
    train_dataset = RoadSegmentationDataset(os.path.join(dataset_path, 'Train/images'), 
                                            os.path.join(dataset_path, 'Train/labels'), 
                                            batch_size, (256, 256))
    val_dataset = RoadSegmentationDataset(os.path.join(dataset_path, 'Validation/images'), 
                                          os.path.join(dataset_path, 'Validation/labels'), 
                                          batch_size, (256, 256))
    

    model = unet_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, update_freq='batch')

    model.fit(train_dataset, 
              epochs=epochs, 
              validation_data=val_dataset, 
              callbacks=[tensorboard_callback])
    
    model.save('unet_road_segmentation.keras')

if __name__ == '__main__':
    train('Image-Segmentation/Data', epochs=1, batch_size=12, learning_rate=0.001)
