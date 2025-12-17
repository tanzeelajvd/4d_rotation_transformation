import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_pca_comparison(X_original, X_transformed, title):
    pca = PCA(n_components=2)
    pca.fit(X_original)

    X_orig_2d = pca.transform(X_original)
    X_rot_2d = pca.transform(X_transformed)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(X_orig_2d[:, 0], X_orig_2d[:, 1], s=10, alpha=0.6)
    plt.title("Original (PCA view)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.subplot(1, 2, 2)
    plt.scatter(X_rot_2d[:, 0], X_rot_2d[:, 1], s=10, alpha=0.6)
    plt.title("Transformed (PCA view)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
