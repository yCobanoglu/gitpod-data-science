import numpy as np
from keras.src.datasets import mnist
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

input = 784
epochs = 1000
latent = 256
learning_rate = 0.1
Im = np.eye(input)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
standardize = StandardScaler().fit(X_train)

X_train = standardize.transform(X_train)
X_test = standardize.transform(X_test)
X_test = X_test.T

X_train = X_train
# code is copied therefore we are doing transpose to reuse gradient formulas Df_W1 and Df_W2
X = X_train.T


XXt = X @ X.T

W1 = np.random.normal(size=(latent, input)) / np.sqrt(input)
W2 = np.random.normal(size=(input, latent)) / np.sqrt(latent)


def loss_fn(X, W1, W2):
    return ((X - W2 @ W1 @ X) ** 2).mean()


def Df_W1(XXt, W1, W2):
    return (W2.T @ (W2 @ W1 - Im)) @ XXt


def Df_W2(XXt, W1, W2):
    return ((W2 @ W1 - Im) @ XXt) @ W1.T


epochs = tqdm(range(epochs))
for epoch in epochs:
    # for mse don't forget to multiply with constant factor (2/ num_of_points_in_batch) for correct gradient
    factor = 2 / (X.shape[0] * X.shape[1])
    W1 -= learning_rate * Df_W1(XXt, W1, W2) * factor
    W2 -= learning_rate * Df_W2(XXt, W1, W2) * factor
    loss = loss_fn(X, W1, W2)
    epochs.set_description(f"Loss: {loss:.4f}")


test_loss = loss_fn(X_test, W1, W2)
print("Test loss: ", test_loss)
