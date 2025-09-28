# main.py
# Orchestrates: dataset load/prepare, model build/compile, training, plotting, and sample predictions.

import tensorflow as tf
import matplotlib.pyplot as plt

from data_loading_and_preprocessing import (
    load_oxford_pet,
    prepare_datasets,
    display_sample,
    IMAGE_SIZE,
    BATCH_SIZE,
)
from model import build_unet_model

# ==== Config ====
OUTPUT_CHANNELS = 3
EPOCHS = 20

def create_mask(pred_mask):
    """Convert softmax logits to a single-channel mask (argmax)."""
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(model, dataset, num=1):
    """Show a few (image, true_mask, predicted_mask) triplets."""
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image, verbose=0)
        display_sample([image[0], mask[0], create_mask(pred_mask)])

def plot_history(history):
    """Plot train/val accuracy and loss."""
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'], loc='upper left')

    plt.tight_layout()
    plt.show()

def main():
    # 1) Load dataset + info
    dataset, info = load_oxford_pet(with_info=True)

    # 2) Prepare datasets
    train_dataset, test_dataset = prepare_datasets(dataset, batch_size=BATCH_SIZE)

    # 3) Build & compile model
    model = build_unet_model(output_channels=OUTPUT_CHANNELS, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 4) Train
    steps_per_epoch = info.splits['train'].num_examples // BATCH_SIZE
    validation_steps = info.splits['test'].num_examples // BATCH_SIZE

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=validation_steps
    )

    # 5) Visualize training curves
    plot_history(history)

    # 6) Show a few predictions
    show_predictions(model, test_dataset, num=10)

if __name__ == "__main__":
    main()
