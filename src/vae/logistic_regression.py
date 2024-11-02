from datetime import datetime

import jax
import jax.numpy as jnp
import keras
import optax
import orbax.checkpoint
import tensorflow as tf
import tensorflow_datasets as tfds
from clu import metrics
from flax import struct
from flax.training import orbax_utils
from flax.training import train_state
from jax import random
from tqdm import tqdm
import os
import numpy as np

from src.flax_autoencoder.model import Autoencoder

jax.config.update("jax_debug_nans", True)

CKPT_NAME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
BATCH_SIZE = 256
WORKERS = 8
LEARNING_RATE = 0.001
EPOCHS = 100
SEED = 42
RESIZE = 240
INPUT_SIZE = RESIZE**2
CKPT_PATH = "src/flax_autoencoder/flax-ckpt/" + CKPT_NAME
CACHE_FILE = "./mean_variance.npz"


################## Load MNIST train and test datasets into memory ##################

TRAIN_SPLIT = 'train[:80%]'  # 80% for training
TEST_SPLIT = 'train[80%:]'   # 20% for testing

# Load the train and test splits separately
train_ds = tfds.load("cats_vs_dogs", split=TRAIN_SPLIT, shuffle_files=True)
test_ds = tfds.load("cats_vs_dogs", split=TEST_SPLIT, shuffle_files=True)

def process_image(data):
    # Convert to grayscale
    data["image"] = tf.image.rgb_to_grayscale(data["image"])
    # Resize to 240x240
    data["image"] = tf.image.resize(data["image"], [RESIZE, RESIZE])
    # Flatten the image
    data["image"] = tf.reshape(data["image"], [-1])  # Flatten to a 1D tensor
    return data


if not os.path.exists(CACHE_FILE):

    _train_dataloader  = train_ds.map(process_image).batch(1).prefetch(tf.data.AUTOTUNE)


    mean_sum = tf.zeros(INPUT_SIZE, dtype=tf.float32)
    squared_diff_sum = tf.zeros(INPUT_SIZE, dtype=tf.float32)
    count = 0

    # First pass: Calculate the mean vector
    for batch in tqdm(_train_dataloader, desc="Calculating Mean"):
        image = batch["image"]
        mean_sum += tf.reduce_sum(image, axis=0)
        count += 1

    mean_vector = mean_sum / tf.cast(count, tf.float32)

    # Second pass: Calculate the variance vector
    for batch in tqdm(_train_dataloader, desc="Calculating Variance"):
        image = batch["image"]
        squared_diff_sum += tf.reduce_sum((image - mean_vector) ** 2, axis=0)

    variance_vector = squared_diff_sum / tf.cast(count, tf.float32)

    # Finalize mean and variance vectors, replacing zero variances with 1
    mean = tf.constant(mean_vector, dtype=tf.float32)
    variance = tf.where(variance_vector == 0, 1.0, variance_vector)
    np.savez(CACHE_FILE, mean_vector=mean_vector.numpy(), variance_vector=variance_vector.numpy())
else:
    cache = np.load(CACHE_FILE)
    mean = tf.constant(cache["mean_vector"], dtype=tf.float32)
    variance = tf.constant(cache["variance_vector"], dtype=tf.float32)

if BATCH_SIZE == -1:
    train_batch_size = train_ds.cardinality()
    test_batch_size = test_ds.cardinality()
else:
    train_batch_size, test_batch_size = BATCH_SIZE, BATCH_SIZE

# Define a function to standardize test data using precomputed mean and adjusted variance
def standardize(data):
    # Convert to grayscale and resize
    image = data["image"]
    standardized_image = (image - mean) / tf.sqrt(variance)
    data["image"] = standardized_image  
    return data

train_dataloader  = train_ds.map(process_image).map(standardize).batch(train_batch_size).prefetch(tf.data.AUTOTUNE)
test_dataloader = train_ds.map(process_image).map(standardize).batch(test_batch_size).prefetch(tf.data.AUTOTUNE)


########################################################################

num_steps_per_epoch = train_dataloader.cardinality().numpy() // EPOCHS
epochs_iterator = tqdm(enumerate(train_dataloader.as_numpy_iterator()), total=train_dataloader.cardinality().numpy())

for step, batch in epochs_iterator:
    pass