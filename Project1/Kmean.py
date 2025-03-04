from sklearn.cluster import KMeans

class KMean:

    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.model = None
    
    def fit(self, X):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        return kmeans.fit_predict(X)