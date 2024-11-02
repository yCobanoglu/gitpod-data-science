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
layers = [256]
nonlinearity = "identity"  # "relu" or "identity"
input_size = 784
ckpt_path = "/home/yunus/PycharmProjects/DeepLearning1/src/flax_autoencoder/flax/" + ckpt_name


################## Load MNIST train and test datasets into memory ##################
train_ds = tfds.load("mnist", split="train", shuffle_files=True)
test_ds = tfds.load("mnist", split="test")


asd = iter(train_ds.batch(len(train_ds)))
_X_train = next(iter(train_ds.batch(len(train_ds))))["image"]
_X_train = tf.reshape(_X_train, [_X_train.shape[0], -1])

_X_test = next(iter(test_ds.batch(len(train_ds))))["image"]
_X_test = tf.reshape(_X_test, [_X_test.shape[0], -1])

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
########################################################################


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(model, rng, learning_rate, input_size):
    """Instanciate the state outside of the training loop"""
    params = model.init(rng, jnp.ones([1, input_size]))["params"]
    opti = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=opti, metrics=Metrics.empty())


@jax.jit
def train_step(state, batch):
    """Train for a single step"""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        loss = optax.squared_error(logits, batch["image"]).mean()
        return loss, logits

    # Update parameters with gradient descent
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    # Update loss, stored in the state object
    metric_updates = state.metrics.single_from_model_output(logits=logits, labels=batch["image"], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)

    return state


@jax.jit
def eval_step(state, batch):
    """Computes the metric on the test batch (code already included in train_step for train batch)"""
    logits = state.apply_fn({"params": state.params}, batch["image"])
    loss = optax.squared_error(logits, batch["image"]).mean()

    metric_updates = state.metrics.single_from_model_output(logits=logits, labels=batch["image"], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state, loss


init_rng = random.PRNGKey(seed)

autoencoder = Autoencoder([input_size, *layers], nonlinearity=nonlinearity)
print(autoencoder.tabulate(jax.random.key(0), jnp.ones((1, input_size)), compute_flops=True, compute_vjp_flops=True))
state = create_train_state(autoencoder, init_rng, learning_rate, input_size)
del init_rng


test_description = ""
train_loss, test_loss = [], []
num_steps_per_epoch = train_dataloader.cardinality().numpy() // epochs
epochs_iterator = tqdm(enumerate(train_dataloader.as_numpy_iterator()), total=train_dataloader.cardinality().numpy())

for step, batch in epochs_iterator:
    state = train_step(state, batch)
    if (step + 1) % num_steps_per_epoch == 0:
        train_loss.append(state.metrics.compute()["loss"].item())
        state = state.replace(metrics=state.metrics.empty())
        epochs_iterator.set_description(
            f"Epoch: {(step + 1) // batch_size}/{epochs} - Train Loss: {train_loss[-1]:.4f}" + test_description
        )
    if (step + 1) % (num_steps_per_epoch * 10) == 0:
        for batch in test_dataloader.as_numpy_iterator():
            state, loss = eval_step(state, batch)
            pass
        test_loss.append(state.metrics.compute()["loss"].item())
        state = state.replace(metrics=state.metrics.empty())
        test_description = f" - Test loss: {test_loss[-1]:.4f}"

ckpt = {
    "model": state,
    "loss_history": {"training": train_loss, "test": test_loss},
    "hyperparameters": {
        "lr": learning_rate,
        "batch_size": batch_size,
        "layers": layers,
        "nonlinearity": nonlinearity,
        "input_size": input_size,
    },
}
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
orbax_checkpointer.save(ckpt_path, ckpt, save_args=save_args)
