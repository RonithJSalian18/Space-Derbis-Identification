"""
Harsh Space-Domain Data Augmentation Pipeline for Space Debris Identification.

Implements graph-compatible TensorFlow augmentation functions simulating harsh orbital environments:
1. Extreme Solar Glare: Localized intense light blooms (direct sun exposure without atmosphere).
2. Sensor Radiation Noise: Salt-and-Pepper / Poisson noise simulating cosmic ray strikes.
3. Photometric Jitter: Aggressive randomized brightness and contrast shifts.
"""

import tensorflow as tf


@tf.function
def add_solar_glare(image: tf.Tensor, glare_prob: float = 0.5, max_intensity: float = 0.8) -> tf.Tensor:
    """
    Randomly adds an intense, localized light bloom simulating direct solar glare in orbit.
    Graph-compatible TensorFlow implementation.
    """
    if tf.random.uniform([]) > glare_prob:
        return image

    shape = tf.shape(image)
    h, w, c = shape[0], shape[1], shape[2]
    h_float = tf.cast(h, tf.float32)
    w_float = tf.cast(w, tf.float32)

    # Random center position for solar bloom
    center_y = tf.random.uniform([], 0.1 * h_float, 0.9 * h_float)
    center_x = tf.random.uniform([], 0.1 * w_float, 0.9 * w_float)
    radius = tf.random.uniform([], 0.15 * h_float, 0.45 * h_float)
    intensity = tf.random.uniform([], 0.4, max_intensity)

    y_grid = tf.range(0, h_float, dtype=tf.float32)[:, tf.newaxis]
    x_grid = tf.range(0, w_float, dtype=tf.float32)[tf.newaxis, :]

    dist_sq = tf.square(y_grid - center_y) + tf.square(x_grid - center_x)
    glare_mask = tf.exp(-dist_sq / (2.0 * tf.square(radius))) * intensity
    glare_mask = tf.expand_dims(glare_mask, axis=-1)

    if c == 3:
        glare_mask = tf.tile(glare_mask, [1, 1, 3])

    augmented = tf.clip_by_value(image + glare_mask, 0.0, 1.0)
    return augmented


@tf.function
def add_sensor_noise(image: tf.Tensor, noise_prob: float = 0.5, salt_pepper_ratio: float = 0.03) -> tf.Tensor:
    """
    Injects random Salt-and-Pepper radiation noise simulating cosmic ray strikes on space sensors.
    Graph-compatible TensorFlow implementation.
    """
    if tf.random.uniform([]) > noise_prob:
        return image

    shape = tf.shape(image)
    random_noise = tf.random.uniform(shape, 0.0, 1.0)

    # Salt noise (white pixels = 1.0)
    salt_mask = tf.cast(random_noise < (salt_pepper_ratio / 2.0), tf.float32)
    # Pepper noise (black pixels = 0.0)
    pepper_mask = tf.cast(random_noise > (1.0 - salt_pepper_ratio / 2.0), tf.float32)

    noisy_image = image * (1.0 - pepper_mask) + salt_mask
    return tf.clip_by_value(noisy_image, 0.0, 1.0)


@tf.function
def apply_photometric_jitter(image: tf.Tensor, jitter_prob: float = 0.5) -> tf.Tensor:
    """
    Aggressive randomized brightness and contrast shifts for orbital illumination variation.
    Graph-compatible TensorFlow implementation.
    """
    if tf.random.uniform([]) > jitter_prob:
        return image

    x = tf.image.random_brightness(image, max_delta=0.3)
    x = tf.image.random_contrast(x, lower=0.5, upper=1.8)
    return tf.clip_by_value(x, 0.0, 1.0)


@tf.function
def add_random_cutout(image: tf.Tensor, cutout_prob: float = 0.5, mask_size_fraction: float = 0.25) -> tf.Tensor:
    """
    Randomly erases a rectangular patch from the image during training to prevent global silhouette or
    border-seam shortcut learning, forcing the CNN to learn localized physical structures (solar panels, edges).
    Graph-compatible TensorFlow implementation.
    """
    if tf.random.uniform([]) > cutout_prob:
        return image

    shape = tf.shape(image)
    h, w, c = shape[0], shape[1], shape[2]
    h_float = tf.cast(h, tf.float32)
    w_float = tf.cast(w, tf.float32)

    cutout_h = tf.cast(h_float * mask_size_fraction, tf.int32)
    cutout_w = tf.cast(w_float * mask_size_fraction, tf.int32)

    y_min = tf.random.uniform([], 0, h - cutout_h, dtype=tf.int32)
    x_min = tf.random.uniform([], 0, w - cutout_w, dtype=tf.int32)

    padding_top = y_min
    padding_bottom = h - (y_min + cutout_h)
    padding_left = x_min
    padding_right = w - (x_min + cutout_w)

    ones_cutout = tf.ones([cutout_h, cutout_w, c], dtype=tf.float32)
    padded_mask = tf.pad(
        ones_cutout,
        [[padding_top, padding_bottom], [padding_left, padding_right], [0, 0]],
        mode='CONSTANT',
        constant_values=0.0
    )

    keep_mask = 1.0 - padded_mask
    return image * keep_mask


@tf.function
def apply_space_domain_augmentations(image: tf.Tensor) -> tf.Tensor:
    """
    Combines solar glare, sensor radiation noise, photometric jitter, random cutout, and zero-g flips/rotations.
    Guarantees returning tensor matching shape (224, 224, C).
    """
    x = image
    x = apply_photometric_jitter(x, jitter_prob=0.6)
    x = add_solar_glare(x, glare_prob=0.5)
    x = add_sensor_noise(x, noise_prob=0.4)
    x = add_random_cutout(x, cutout_prob=0.4, mask_size_fraction=0.25)

    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    k_rot = tf.random.uniform([], 0, 4, dtype=tf.int32)
    x = tf.image.rot90(x, k=k_rot)

    return x


def get_data_augmentation_pipeline(image_size: tuple = (224, 224)) -> tf.keras.Sequential:
    """
    Builds a Keras Sequential pipeline incorporating harsh space-domain augmentations.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Lambda(lambda img: apply_space_domain_augmentations(img), name="space_domain_aug"),
        ],
        name="space_data_augmentation",
    )
