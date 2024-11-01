############# Flax model ###########################
import jax
from flax import linen as nn
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import random
from keras.src.datasets import mnist
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

jax.config.update("jax_debug_nans", True)

input_size = 784
epochs = 100
latent = 256
learning_rate = 0.001
seed = 42

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
standardize = StandardScaler().fit(X_train)

X_train = standardize.transform(X_train)
X_test = standardize.transform(X_test)


def create_train_state(model, rng, learning_rate, input_size):
    """Instanciate the state outside of the training loop"""
    params = model.init(rng, jnp.ones([1, input_size]))["params"]
    opti = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opti)


@jax.jit
def loss_fn(params, X):
    logits = state.apply_fn({"params": params}, X)
    loss = optax.squared_error(logits, X).mean()
    return loss


@jax.jit
def train_step(state, X):
    """Train for a single step"""
    # Update parameters with gradient descent
    loss, grads = jax.value_and_grad(loss_fn)(state.params, X)
    state = state.apply_gradients(grads=grads)
    return loss, state


class Autoencoder(nn.Module):
    latent: int

    @nn.compact
    def __call__(self, x):
        x1 = nn.Dense(features=self.latent, use_bias=False)(x)
        return nn.Dense(features=x.shape[1], use_bias=False)(x1)


autoencoder = Autoencoder(latent)
init_rng = random.PRNGKey(seed)
state = create_train_state(autoencoder, init_rng, learning_rate, input_size)
del init_rng

epochs_iterator = tqdm(range(epochs))
test_descr = ""
for epoch in epochs_iterator:
    loss, state = train_step(state, X_train)
    if epoch % 10 == 0 and epoch != 0:
        test_loss = loss_fn(state.params, X_test)
        test_descr = f"- Test Loss:{test_loss:.4f}"
    epochs_iterator.set_description(f"Epoch: {epoch + 1}/{epochs} - Train Loss: {loss:.4f}" + test_descr)
