import matplotlib
import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def display_images_and_reconstructed_images(recon, imgs):
    n_images = 10
    plt.figure(figsize=(20, 4))
    for i in range(n_images):
        # Original image
        ax = plt.subplot(2, n_images, i + 1)
        plt.imshow(recon[i].reshape(28, 28), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Reconstructed image
        ax = plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(imgs[i].reshape(28, 28), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def display_latent_space(latent, labels, dim=2):
    assert dim in [2, 3], "Only 2D and 3D visualizations are supported"
    if dim == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latent[:, 0], latent[:, 1], c=labels, cmap="viridis", s=1)
        plt.colorbar(scatter)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(f"PCA of MNIST Dataset {dim}D")
        plt.show()
    if dim == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], c=labels, cmap="viridis", s=1)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.set_title("PCA of MNIST Dataset (3D)")
        fig.colorbar(scatter)
        plt.show()


def kmeans(train_samples, test_samples):
    X_train, X_train_labels = train_samples
    X_test, X_test_labels = test_samples

    # get unique num of labels from X_train_labels
    clusters = len(np.unique(X_train_labels))
    print("Number of clusters for kmeans: ", clusters)
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(X_train)
    train_clusters = kmeans.predict(X_train)

    test_clusters = kmeans.predict(X_test)
    cluster_to_label = {}  # Initialize an empty dictionary to store cluster to label mappings

    for cluster in range(10):  # Loop over all 10 clusters (assuming 10 clusters for 10 digits)
        mask = train_clusters == cluster  # Create a boolean mask for all samples in the current cluster
        if np.any(mask):  # Check if there are any samples in this cluster
            common_label = mode(X_train_labels[mask]).mode  # Find the most frequent label in this cluster
            cluster_to_label[cluster] = common_label

    # Map test clusters to labels
    y_test_pred = np.array([cluster_to_label[cluster] for cluster in test_clusters])

    # Calculate accuracy
    return accuracy_score(X_test_labels, y_test_pred)
