import jax

jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import matplotlib
from sklearn.cluster import KMeans

from src.utils import display_images_and_reconstructed_images, display_latent_space

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from keras.src.datasets import mnist
from sklearn.preprocessing import StandardScaler

"""PCA on MNIST dataset"""
CUT_OFF = 256
data_path = "/data"

########### Dataset ##############################################
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

standardize = StandardScaler().fit(X_train)

X_train = standardize.transform(X_train)
print("Mean after standardization: ", X_train.mean())
print("Variance after standardization: ", X_train.var())
X_test = standardize.transform(X_test)
##################################################################

U, S, Vt = jnp.linalg.svd(X_train, full_matrices=False)

# plot the singular values (to visually choose cut off)
plt.figure(figsize=(8, 6))
plt.plot(S)
plt.yscale("log")
plt.xlabel("Singular Value Index")
plt.ylabel("Singular Value (log scale)")
plt.title("Singular Values of MNIST Data")
plt.show()

principal_components = Vt[:CUT_OFF].T  # Transpose to get principal components

# Project the training data onto the principal components
X_train_pca = X_train @ principal_components

# Project the test data onto the principal components
X_test_pca = X_test @ principal_components

# Reconstruct the train data from the PCA representation
X_train_reconstructed = X_train_pca @ principal_components.T

# Reconstruct the test data from the PCA representation
X_test_reconstructed = X_test_pca @ principal_components.T

# Compute the MSE error on the test set
mse_error = lambda X, Y: ((X - Y) ** 2).mean()

print(f"MSE error on the train set: {mse_error(X_train, X_train_reconstructed):.4f}")
print(f"MSE error on the test set: {mse_error(X_test, X_test_reconstructed):.4f}")


# Show the reconstructed test images
X_test_undo_preprocessing = standardize.inverse_transform(X_test)
X_test_reconstructed_undo_preprocessing = standardize.inverse_transform(X_test_reconstructed)


display_images_and_reconstructed_images(
    standardize.inverse_transform(X_test_reconstructed), standardize.inverse_transform(X_test)
)


display_latent_space(X_train_pca, y_train, dim=2)
display_latent_space(X_train_pca, y_train, dim=3)

# Perform k-means clustering on the training set
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train_pca)
train_clusters = kmeans.predict(X_train_pca)

# Predict clusters for the test set
test_clusters = kmeans.predict(X_test_pca)

cluster_to_label = {}  # Initialize an empty dictionary to store cluster to label mappings

for cluster in range(10):  # Loop over all 10 clusters (assuming 10 clusters for 10 digits)
    mask = train_clusters == cluster  # Create a boolean mask for all samples in the current cluster
    if jnp.any(mask):  # Check if there are any samples in this cluster
        common_label = mode(y_train[mask]).mode  # Find the most frequent label in this cluster
        cluster_to_label[cluster] = common_label

# Map test clusters to labels
y_test_pred = jnp.array([cluster_to_label[cluster] for cluster in test_clusters])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Clustering accuracy on the test set: {accuracy:.4f}")
