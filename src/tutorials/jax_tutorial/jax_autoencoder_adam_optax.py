import jax
import optax
import jax.numpy as np
from jax import random
from jax import value_and_grad
from keras.src.datasets import mnist
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

jax.config.update("jax_debug_nans", True)

input = 784
epochs = 1000
latent = 256
learning_rate = 0.001
seed = 42

optimizer = optax.adam(learning_rate)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
standardize = StandardScaler().fit(X_train)

X_train = standardize.transform(X_train)
X_test = standardize.transform(X_test)

key = random.PRNGKey(seed)
k1, k2 = random.split(key)
W1 = random.normal(k1, (input, latent)) / np.sqrt(input)
W2 = random.normal(k2, (latent, input)) / np.sqrt(latent)
params = (W1, W2)

opt_state = optimizer.init(params)


@jax.jit
def loss_fn(params, X):
    W1, W2 = params
    return optax.squared_error(X @ W1 @ W2, X).mean()


@jax.jit
def train_step(params, opt_state):
    loss, grads = value_and_grad(loss_fn)(params, X_train)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params


opt_step = 0
epochs = tqdm(range(epochs))
test_loss_descr = ""
for epoch in epochs:
    loss, grads = value_and_grad(loss_fn)(params, X_train)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if epoch % 10 == 0 and epoch > 0:
        test_loss = loss_fn(params, X_test)
        test_loss_descr = f" Test loss: {test_loss:.4f}"
    epochs.set_description(f"Loss: {loss:.4f}" + test_loss_descr)
