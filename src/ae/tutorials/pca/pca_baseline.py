from keras.src.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils import display_images_and_reconstructed_images, display_latent_space, kmeans

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

N_COMPONENTS = 512
pca = PCA(n_components=N_COMPONENTS)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

X_train_reconstructed = pca.inverse_transform(X_train_pca)
X_test_reconstructed = pca.inverse_transform(X_test_pca)


def mse(X, Y):
    return ((X - Y) ** 2).mean()


print(f"MSE on the training set: {mse(X_train, X_train_reconstructed):.4f}")
print(f"MSE on the test set: {mse(X_test, X_test_reconstructed):.4f}")

display_images_and_reconstructed_images(
    standardize.inverse_transform(X_test_reconstructed), standardize.inverse_transform(X_test)
)


display_latent_space(X_train_pca, y_train, dim=2)
display_latent_space(X_train_pca, y_train, dim=3)

print("Accuracy using kmeans :", kmeans((X_train_pca, y_train), (X_test_pca, y_test)))
