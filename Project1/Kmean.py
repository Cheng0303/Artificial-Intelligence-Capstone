from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder

class KMean:

    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.model = None
    
    def fit(self, X, y=None):
        # Apply PCA for Dimensionality Reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.toarray())  # Convert sparse to dense array for PCA

        # Apply KMeans Clustering
        num_clusters = 3  # Set number of clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_pca)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        ari = adjusted_rand_score(y_encoded, cluster_labels)
        nmi = normalized_mutual_info_score(y_encoded, cluster_labels)
        print(f"Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

        # Accuracy (assuming labels match cluster assignment)
        accuracy = sum(y_encoded == cluster_labels) / len(y_encoded)
        print(f"Accuracy: {accuracy:.4f}")

        # Silhouette Score
        silhouette = silhouette_score(X, cluster_labels)
        print(f"Silhouette Score: {silhouette:.4f}")

        # Inertia (Sum of Squared Errors)
        sse = kmeans.inertia_
        print(f"Inertia (SSE): {sse:.4f}")


        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette="viridis", s=10, edgecolor="k")

        # Plot Cluster Centers
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                    s=300, c="red", marker="X", label="Centroids")

        plt.title("K-Means Clustering Visualization")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.savefig("KMean.png")
        plt.show()

        return cluster_labels