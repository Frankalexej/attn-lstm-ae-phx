import numpy as np
from sklearn.cluster import DBSCAN, KMeans

class DirectionalConsistency: 
    @staticmethod
    def analyze(trajectory): 
        if trajectory.shape[0] <= 2: 
            return None, 0.
        
        # Compute direction vectors between consecutive points
        direction_vectors = np.diff(trajectory, axis=0)

        # Calculate angles between consecutive direction vectors
        angles = []
        for i in range(1, len(direction_vectors)):
            v1 = direction_vectors[i - 1]
            v2 = direction_vectors[i]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angles.append(angle)

        # Calculate variance of the angles
        variance = np.var(angles)

        return angles, variance
    


class Clusterer: 
    @staticmethod
    def dbscan(data, r=0.36, m=1): 
        # Create a DBSCAN instance
        dbscan = DBSCAN(eps=r, min_samples=m)  # Adjust parameters as needed

        # Fit the DBSCAN model to your data
        dbscan.fit(data)

        # Get cluster labels (-1 represents noise/outliers)
        labels = dbscan.labels_

        # Get the number of clusters (excluding noise points)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return labels, num_clusters
    

    @staticmethod
    def kmeans(data, n_clusters=1): 
        if data.shape[0] < n_clusters:
            # strange cases when number of clusters exceeds numebr of points
            # in this case just give each point one cluster label
            return np.array(list(range(data.shape[0])), dtype=np.int32), data 

        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init='auto')
        kmeans.fit(data)
        # Get cluster assignments for each data point
        cluster_assignments = kmeans.labels_
        # Get cluster centroids
        cluster_centers = kmeans.cluster_centers_
        return cluster_assignments, cluster_centers