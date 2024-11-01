import jax.numpy as jnp
import keras
import orbax.checkpoint as ocp
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import random

from src.flax_autoencoder.model import Autoencoder
from src.utils import display_images_and_reconstructed_images, display_latent_space, kmeans

orbax_checkpointer = ocp.PyTreeCheckpointer()
path = "/home/yunus/PycharmProjects/DeepLearning1/src/flax_autoencoder/flax/2024-07-19 14:50:06"
raw_restored = orbax_checkpointer.restore(path)


input_size = raw_restored["hyperparameters"]["input_size"]
layers = raw_restored["hyperparameters"]["layers"]
nonlinearity = raw_restored["hyperparameters"]["nonlinearity"]
params = raw_restored["model"]["params"]


################## Load MNIST train and test datasets into memory ##################
train_ds = tfds.load("mnist", split="train", shuffle_files=True)
test_ds = tfds.load("mnist", split="test")


def my_transform(sample):
    # unable to debug this https://stackoverflow.com/questions/59275095/ide-breakpoint-in-tensorflow-dataset-api-mapped-py-function/59346218#59346218
    image = sample["image"]
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [-1]) / 255
    return {"image": image, "label": sample["label"]}


train_ds = train_ds.map(my_transform)
test_ds = test_ds.map(my_transform)

train_dataloader = train_ds.batch(len(train_ds)).as_numpy_iterator()
test_dataloader = test_ds.batch(len(test_ds)).as_numpy_iterator()
X_train, X_train_labels = next(train_dataloader).values()
X_test, X_test_labels = next(test_dataloader).values()
########################################################################

seed = 42
init_rng = random.PRNGKey(seed)
model = Autoencoder([input_size, *layers], nonlinearity=nonlinearity)


X_test_reconstructed = model.apply({"params": params}, X_test)

display_images_and_reconstructed_images(X_test_reconstructed, X_test)


X_train_latent = model.apply({"params": params}, X_train, method=model.encode)
X_test_latent = model.apply({"params": params}, X_test, method=model.encode)


display_latent_space(X_train_latent, X_train_labels, dim=2)
display_latent_space(X_train_latent, X_train_labels, dim=3)

print("Accuracy using kmeans :", kmeans((X_train_latent, X_train_labels), (X_test_latent, X_test_labels)))
