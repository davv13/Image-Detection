import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model

def conv_block(input_tensor, num_filters):
    """An encoder block with two convolutional layers."""
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
    return x

def encoder_block(input_tensor, num_filters):
    """An encoder block with a convolution block followed by max pooling."""
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, concat_tensor, num_filters):
    """A decoder block."""
    x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    x = concatenate([x, concat_tensor])
    x = conv_block(x, num_filters)
    return x

def unet_model(input_size=(256, 256, 3), num_filters=64, num_classes=1):
    """Build a U-Net model."""
    inputs = Input(input_size)

    # Encoder
    c1, p1 = encoder_block(inputs, num_filters)
    c2, p2 = encoder_block(p1, num_filters*2)
    c3, p3 = encoder_block(p2, num_filters*4)
    c4, p4 = encoder_block(p3, num_filters*8)

    # Bridge
    b = conv_block(p4, num_filters*16)

    # Decoder
    d1 = decoder_block(b, c4, num_filters*8)
    d2 = decoder_block(d1, c3, num_filters*4)
    d3 = decoder_block(d2, c2, num_filters*2)
    d4 = decoder_block(d3, c1, num_filters)

    # Output
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(d4)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def compile_model(model, lr=0.001):
    """Compile the U-Net model."""
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',  # or 'categorical_crossentropy'
                  metrics=['accuracy'])
    return model
