import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def extract_features(point_cloud):
    """
    Extract geometric features from a point cloud for defect detection.
    
    Args:
        point_cloud (dict): Input point cloud dictionary with 'points', 'colors', 'normals' keys
        
    Returns:
        numpy.ndarray: Feature matrix
        list: Feature names
    """
    points = point_cloud['points']
    normals = point_cloud['normals'] if 'normals' in point_cloud and point_cloud['normals'] is not None else None
    
    # Initialize features list
    features = []
    feature_names = []
    
    # Basic position features
    features.append(points)
    feature_names.extend(['x', 'y', 'z'])
    
    # If normals are available, use them as features
    if normals is not None:
        features.append(normals)
        feature_names.extend(['nx', 'ny', 'nz'])
    
    # Color features if available
    if 'colors' in point_cloud and point_cloud['colors'] is not None:
        colors = point_cloud['colors']
        features.append(colors)
        feature_names.extend(['r', 'g', 'b'])
    
    # Local point density (using KNN)
    if len(points) > 10:
        nbrs = NearestNeighbors(n_neighbors=10).fit(points)
        distances, _ = nbrs.kneighbors(points)
        avg_distances = np.mean(distances, axis=1).reshape(-1, 1)
        features.append(avg_distances)
        feature_names.append('local_density')
    
    # Roughness estimation (if normals are available)
    if normals is not None and len(points) > 10:
        nbrs = NearestNeighbors(n_neighbors=10).fit(points)
        _, indices = nbrs.kneighbors(points)
        
        roughness = np.zeros((len(points), 1))
        for i in range(len(points)):
            neighbor_normals = normals[indices[i]]
            # Compute variance of normals as a roughness measure
            normal_var = np.var(neighbor_normals, axis=0)
            roughness[i] = np.sum(normal_var)
        
        features.append(roughness)
        feature_names.append('roughness')
    
    # Concatenate all features
    feature_matrix = np.hstack(features)
    
    return feature_matrix, feature_names


def detect_defects_features(point_cloud, model_type="Random Forest", feature_set=None):
    """
    Detect defects in a point cloud using geometric features and classification.
    
    This function extracts features from the point cloud and uses a machine learning
    model to classify points as defects or normal structure.
    
    Args:
        point_cloud (dict): Input point cloud dictionary with 'points', 'colors', 'normals' keys
        model_type (str): Type of model to use ("Random Forest", "SVM", "Isolation Forest")
        feature_set (list): List of features to use (subset of ["Normals", "Curvature", "Roughness", "Verticality"])
        
    Returns:
        numpy.ndarray: Labels for each point (1 for defect, 0 for normal)
        numpy.ndarray: Indices of points classified as defects
        dict: Feature importance information (features and importance values)
        str: Classification report
    """
    # Extract features from point cloud
    X, feature_names = extract_features(point_cloud)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # For demonstration, we'll create a small subset of synthetic defects
    # In a real application, this would be replaced by an actual ML model
    
    n_samples = len(point_cloud['points'])
    
    if model_type == "Isolation Forest":
        # Isolation Forest for anomaly detection
        model = IsolationForest(contamination=0.05, random_state=42)
        y_pred = model.fit_predict(X_scaled)
        # Convert to binary labels (1 for defects, 0 for normal)
        labels = np.where(y_pred == -1, 1, 0)
        
        # No direct feature importance for Isolation Forest
        feature_importance = np.ones(X.shape[1]) / X.shape[1]
        
    elif model_type == "SVM":
        # For demonstration, generate synthetic labels 
        # In a real application, you would have labeled training data
        y_train = np.zeros(n_samples)
        # Mark the 5% most "unusual" points as defects based on distance from center
        center = np.mean(point_cloud['points'], axis=0)
        distances = np.linalg.norm(point_cloud['points'] - center, axis=1)
        threshold = np.percentile(distances, 95)
        y_train[distances > threshold] = 1
        
        # Train SVM
        model = SVC(kernel='rbf', probability=True)
        model.fit(X_scaled, y_train)
        labels = model.predict(X_scaled)
        
        # SVM doesn't provide feature importance directly
        feature_importance = np.ones(X.shape[1]) / X.shape[1]
        
    else:  # Default: Random Forest
        # For demonstration, generate synthetic labels
        # In a real application, you would have labeled training data
        y_train = np.zeros(n_samples)
        # Mark the 5% most "unusual" points as defects based on distance from center
        center = np.mean(point_cloud['points'], axis=0)
        distances = np.linalg.norm(point_cloud['points'] - center, axis=1)
        threshold = np.percentile(distances, 95)
        y_train[distances > threshold] = 1
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y_train)
        labels = model.predict(X_scaled)
        
        # Get feature importance
        feature_importance = model.feature_importances_
    
    # Find indices of defect points
    defect_indices = np.where(labels == 1)[0]
    
    # Create a classification report
    report = classification_report(y_train, labels, output_dict=True)
    
    # Create feature importance dictionary
    importance_dict = {
        'features': feature_names,
        'importance': feature_importance
    }
    
    return labels, defect_indices, importance_dict, report


def detect_defects_clustering(point_cloud, method="dbscan", eps=0.1, min_samples=10, n_clusters=5):
    """
    Detect defects using clustering algorithms.
    
    Args:
        point_cloud (dict): Input point cloud dictionary with 'points', 'colors', 'normals' keys
        method (str): Clustering method ('dbscan' or 'kmeans')
        eps (float): Epsilon parameter for DBSCAN
        min_samples (int): Minimum samples parameter for DBSCAN
        n_clusters (int): Number of clusters for K-Means
        
    Returns:
        numpy.ndarray: Cluster labels for each point
        numpy.ndarray: Indices of points classified as defects
    """
    # Extract features for clustering
    X, _ = extract_features(point_cloud)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == "dbscan":
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(X_scaled)
        
        # Points labeled as noise (-1) are considered defects
        defect_indices = np.where(labels == -1)[0]
        
    else:  # K-Means
        # K-Means clustering
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clustering.fit_predict(X_scaled)
        
        # Determine which clusters represent defects based on cluster size
        # Smaller clusters are more likely to be defects
        cluster_sizes = np.bincount(labels)
        # Consider clusters that are significantly smaller than the average as defects
        avg_size = np.mean(cluster_sizes)
        defect_clusters = np.where(cluster_sizes < avg_size * 0.5)[0]
        
        # Get indices of points in defect clusters
        defect_indices = np.array([i for i, label in enumerate(labels) if label in defect_clusters])
    
    return labels, defect_indices


def detect_defects_segmentation(point_cloud, method="region_growing", smoothness_threshold=3.0, min_cluster_size=100, distance_threshold=0.05):
    """
    Detect defects using segmentation methods.
    
    Args:
        point_cloud (dict): Input point cloud dictionary with 'points', 'colors', 'normals' keys
        method (str): Segmentation method ('region_growing' or 'ransac')
        smoothness_threshold (float): Smoothness threshold for region growing
        min_cluster_size (int): Minimum cluster size for region growing
        distance_threshold (float): Distance threshold for plane detection
        
    Returns:
        numpy.ndarray: Labels for each point (1 for defect, 0 for normal)
        numpy.ndarray: Indices of points classified as defects
    """
    # Extract features
    points = point_cloud['points']
    normals = point_cloud['normals'] if 'normals' in point_cloud and point_cloud['normals'] is not None else None
    
    # For region growing, we need normals
    if normals is None:
        # If normals not available, estimate them based on local neighborhoods
        # This is a simplified version and not as accurate as Open3D's implementation
        normals = np.zeros_like(points)
        nbrs = NearestNeighbors(n_neighbors=10).fit(points)
        _, indices = nbrs.kneighbors(points)
        
        for i in range(len(points)):
            neighbors = points[indices[i]]
            # Calculate covariance matrix of neighbors
            cov = np.cov(neighbors.T)
            # Get eigenvectors and eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Normal is the eigenvector corresponding to the smallest eigenvalue
            normals[i] = eigenvectors[:, 0]
    
    if method == "region_growing":
        # Simplified region growing implementation
        # Note: A full implementation would be more complex and recursive
        
        # Initialize labels as unassigned (-1)
        labels = np.full(len(points), -1)
        visited = np.zeros(len(points), dtype=bool)
        
        # Use KNN for neighborhood search
        nbrs = NearestNeighbors(n_neighbors=20).fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        current_label = 0
        
        for i in range(len(points)):
            if visited[i]:
                continue
            
            # Start a new region
            seed_queue = [i]
            region = []
            
            while seed_queue:
                current_seed = seed_queue.pop(0)
                if visited[current_seed]:
                    continue
                
                visited[current_seed] = True
                region.append(current_seed)
                
                # Check neighbors
                for neighbor_idx in indices[current_seed]:
                    if not visited[neighbor_idx]:
                        # Calculate angle between normals
                        angle = np.arccos(np.clip(np.dot(normals[current_seed], normals[neighbor_idx]), -1.0, 1.0))
                        angle_deg = np.degrees(angle)
                        
                        # If angle is below threshold, add to region
                        if angle_deg < smoothness_threshold:
                            seed_queue.append(neighbor_idx)
            
            # If region is large enough, assign label
            if len(region) >= min_cluster_size:
                labels[region] = current_label
                current_label += 1
        
        # Points that weren't assigned to any region are considered defects
        defect_indices = np.where(labels == -1)[0]
        
    else:  # RANSAC plane segmentation
        # Simplified RANSAC implementation for plane detection
        # In a real implementation, you would run RANSAC multiple times
        # to find multiple planes
        
        max_iterations = 100
        best_inliers = []
        
        for _ in range(max_iterations):
            # Randomly select 3 points
            sample_indices = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[sample_indices]
            
            # Calculate plane equation Ax + By + Cz + D = 0
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, p1)
            
            # Calculate distances from all points to the plane
            distances = np.abs(np.dot(points, normal) + d)
            
            # Find inliers
            inliers = np.where(distances < distance_threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
        
        # The points that are not part of the dominant plane are considered defects
        labels = np.ones(len(points))
        labels[best_inliers] = 0
        defect_indices = np.where(labels == 1)[0]
    
    return labels, defect_indices