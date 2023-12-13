import os
import model
import dataset
import argparse
import tensorflow as tf

def main(data_dir, model_dir, epochs, batch_size, learning_rate):
    # Load the dataset
    train_dataset = dataset.load_data(data_dir, batch_size=batch_size, image_size=(256, 256), mode='training')
    val_dataset = dataset.load_data(data_dir, batch_size=batch_size, image_size=(256, 256), mode='validation')

    # Initialize the model
    road_segmentation_model = model.unet_model()
    road_segmentation_model = model.compile_model(road_segmentation_model, lr=learning_rate)

    # Setup checkpoint callback
    checkpoint_path = os.path.join(model_dir, "cp-{epoch:04d}.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # Train the model
    history = road_segmentation_model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[cp_callback])

    # Save the final model
    final_model_path = os.path.join(model_dir, 'final_model')
    road_segmentation_model.save(final_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a road segmentation model.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory for the dataset')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory to save the models')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.epochs, args.batch_size, args.lr)
