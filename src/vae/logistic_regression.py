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

from src.flax_autoencoder.model import Autoencoder

# jax.config.update("jax_debug_nans", True)

ckpt_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
batch_size = 1024
workers = 8
learning_rate = 0.001
epochs = 100
seed = 42
input_size = 784
ckpt_path = "src/flax_autoencoder/flax-ckpt/" + ckpt_name


################## Load MNIST train and test datasets into memory ##################
train_ds = tfds.load("cats_vs_dogs", split="train", shuffle_files=True)
print("Test")
O = next(iter(train_ds.batch(len(train_ds))))
print(O)
_X = next(iter(train_ds.batch(len(train_ds))))["image"]
_X = tf.reshape(_X, [_X.shape[0], -1])
pass
"""

standardize = keras.layers.Normalization(axis=-1)
standardize.adapt(_X_train)

# https://github.com/keras-team/keras/issues/20011
variance = standardize.variance
standardize.variance = tf.where(variance == 0, tf.ones_like(variance), variance)

_X_train = standardize(_X_train)
_X_test = standardize(_X_test)

print("Mean on train: ", tf.math.reduce_mean(_X_train).numpy())
print("Var on train: ", tf.math.reduce_variance(_X_train).numpy())

print("Mean on test (fitted using train mean and var): ", tf.math.reduce_mean(_X_test).numpy())
print("Var on test: ", tf.math.reduce_variance(_X_test).numpy())


def my_transform(sample):
    # unable to debug this https://stackoverflow.com/questions/59275095/ide-breakpoint-in-tensorflow-dataset-api-mapped-py-function/59346218#59346218
    image = sample["image"]
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [-1]) / 255
    # image = standardize(image)
    # image = tf.reshape(image, [-1])
    return {"image": image, "label": sample["label"]}


train_ds = train_ds.map(my_transform)
test_ds = test_ds.map(my_transform)

if batch_size == -1:
    train_batch_size = train_ds.cardinality()
    test_batch_size = test_ds.cardinality()
else:
    train_batch_size, test_batch_size = batch_size, batch_size

train_dataloader = (
    train_ds.repeat(epochs).shuffle(1024).batch(train_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
)
test_dataloader = test_ds.batch(test_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

"""
########################################################################