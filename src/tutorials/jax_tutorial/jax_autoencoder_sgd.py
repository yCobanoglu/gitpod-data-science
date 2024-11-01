import jax

jax.config.update("jax_debug_nans", True)
import jax.numpy as np
from jax import random
from jax import value_and_grad
from keras.src.datasets import mnist
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

input = 784
epochs = 100
latent = 256
learning_rate = 1
seed = 42
Im = np.eye(input)

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


@jax.jit
def loss_fn(params, X):
    W1, W2 = params
    return ((X @ W1 @ W2 - X) ** 2).mean()


@jax.jit
def update(params, grads):
    return jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)


epochs = tqdm(range(epochs))
for epoch in epochs:
    loss, grads = value_and_grad(loss_fn)(params, X_train)
    params = update(params, grads)
    epochs.set_description(f"Loss: {loss:.4f}")

test_loss = loss_fn(params, X_test)
print("Test loss: ", test_loss)
