import tensorflow as tf
from tensorflow.keras import layers

def create_unet_model(input_shape, num_classes):
    """
    Create a U-Net segmentation model.

    Parameters:
    - input_shape (tuple): The shape of the input images (height, width, channels).
    - num_classes (int): The number of classes for segmentation.

    Returns:
    - tf.keras.Model: The U-Net segmentation model.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    # pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    up3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv2)
    concat3 = layers.concatenate([conv1, up3], axis=-1)
    conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat3)
    conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(conv3)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
