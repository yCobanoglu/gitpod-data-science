import os
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import random
from tqdm import tqdm



#jax.config.update("jax_debug_nans", True)

CKPT_NAME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
BATCH_SIZE = 256
SAMPLE_BATCH_SIZE = 10
WORKERS = 8
LEARNING_RATE = 0.01
EPOCHS = 100
SEED = 42
RESIZE = 240
INPUT_SIZE = RESIZE**2
CKPT_PATH = "src/flax_autoencoder/flax-ckpt/" + CKPT_NAME
CACHE_FILE = "./mean_variance.npz"


################## Load & preprocess Dataset ##################

TRAIN_SPLIT = 'train[:80%]'  # 80% for training
TEST_SPLIT = 'train[80%:]'   # 20% for testing

# Load the train and test splits separately
train_ds = tfds.load("cats_vs_dogs", split=TRAIN_SPLIT, shuffle_files=True)
test_ds = tfds.load("cats_vs_dogs", split=TEST_SPLIT, shuffle_files=True)

# print cardinality of the datasets
print("Number of total samples in train set:", train_ds.cardinality().numpy())
print("Number of total samples in test set:", test_ds.cardinality().numpy())

def filter_cats(data):
    return data["label"] == 1

# print cardinality of the datasets
print("Number of total cat samples in train set:", sum(1 for _ in train_ds.filter(filter_cats)))
print("Number of total cat samples in test set:", sum(1 for _ in test_ds.filter(filter_cats)))

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
    image = data["image"]
    standardized_image = (image - mean) / tf.sqrt(variance)
    data["image"] = standardized_image  
    return data

train_dataloader  = train_ds.filter(filter_cats).map(process_image).map(standardize).batch(train_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
test_dataloader = test_ds.filter(filter_cats).map(process_image).map(standardize).batch(test_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# Verify the number of samples in the filtered datasets
print("Number of cat images in train set:", sum(1 for _ in tqdm(train_dataloader)) * train_batch_size)
print("Number of cat images in test set:", sum(1 for _ in tqdm(test_dataloader)) * test_batch_size)

########################################################################


################### Display Images ################################
save_dir = 'training_images'
# delete the directory if it already exists
if os.path.exists(save_dir):
    import shutil
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)

# Function to save and display images from a TensorFlow dataset

for batch_num, batch in enumerate(train_dataloader.take(1)):
    image_batch = batch["image"]
    for i in range(max(image_batch.shape[0], 10)):
        image = image_batch[i]
        image = image * tf.sqrt(variance) + mean
        image = tf.reshape(image, [RESIZE, RESIZE])
        image_array = image.numpy() # Convert to numpy and remove channel dimension for grayscale
        
        # Save image
        file_path = os.path.join(save_dir, f"image_batch{batch_num}_img{i}.png")
        plt.imsave(file_path, image_array, cmap='gray')

###################################################################