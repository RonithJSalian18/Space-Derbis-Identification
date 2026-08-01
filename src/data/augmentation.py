import tensorflow as tf


def get_data_augmentation_pipeline(image_size: tuple = (224, 224)) -> tf.keras.Sequential:
    """
    Builds a GPU-accelerated data augmentation sequential pipeline.
    Tailored for zero-g orbital space images (rotations, flips, zoom, brightness).
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.5),  # up to 180 degrees
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="space_data_augmentation",
    )
