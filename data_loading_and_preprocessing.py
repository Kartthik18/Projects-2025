# data_loading_and_preprocessing.py
# Utilities for loading Oxford-IIIT Pet, preprocessing, batching, and simple visualization.

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 64
BUFFER_SIZE = 1000
AUTOTUNE = tf.data.AUTOTUNE

def normalize(input_image, input_mask):
    """Scale image to [0,1] and shift mask labels from {1,2,3} to {0,1,2}."""
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = input_mask - 1  # convert to zero-based indexing
    return input_image, input_mask

def load_train_images(sample):
    """Resize, augment (flip), and normalize."""
    # Resize image (bilinear) and mask (nearest-neighbor to keep class IDs intact)
    input_image = tf.image.resize(sample['image'], IMAGE_SIZE)
    input_mask  = tf.image.resize(sample['segmentation_mask'], IMAGE_SIZE,
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Simple augmentation
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask  = tf.image.flip_left_right(input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def load_test_images(sample):
    """Resize and normalize (no augmentation)."""
    input_image = tf.image.resize(sample['image'], IMAGE_SIZE)
    input_mask  = tf.image.resize(sample['segmentation_mask'], IMAGE_SIZE,
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def prepare_datasets(dataset_dict,
                     batch_size: int = BATCH_SIZE,
                     buffer_size: int = BUFFER_SIZE):
    """
    Takes the dict returned by tfds.load(..., with_info=True)[0] and returns
    (train_dataset, test_dataset) ready for model.fit().
    """
    train_dataset = dataset_dict['train'].map(load_train_images, num_parallel_calls=AUTOTUNE)
    test_dataset  = dataset_dict['test'].map(load_test_images,   num_parallel_calls=AUTOTUNE)

    train_dataset = (
        train_dataset
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
        .repeat()
        .prefetch(buffer_size=AUTOTUNE)
    )
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset

def display_sample(image_list):
    """Quick side-by-side viewer: [input_image, true_mask, (optional) predicted_mask]."""
    plt.figure(figsize=(10, 10))
    titles = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(image_list)):
        plt.subplot(1, len(image_list), i + 1)
        plt.title(titles[i])
        plt.imshow(tf.keras.utils.array_to_img(image_list[i]))
        plt.axis('off')
    plt.show()

def load_oxford_pet(with_info: bool = True):
    """Wrapper to load Oxford-IIIT Pet dataset."""
    dataset, info = tfds.load('oxford_iiit_pet', with_info=with_info)
    return dataset, info
