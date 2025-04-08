import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as sklearn_classification_report
import logging
from preprocessing import extract_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    logger.info(f"Starting defect detection using {model_type} classifier")
    
    # Extract features from the point cloud
    features, feature_names = extract_features(point_cloud)
    
    # Filter features based on feature_set if provided
    if feature_set is not None:
        selected_indices = []
        selected_features = []
        
        # Map feature_set to feature indices
        if "Normals" in feature_set:
            selected_indices.extend([0, 1, 2])
            selected_features.extend(['normal_x', 'normal_y', 'normal_z'])
        if "Curvature" in feature_set and 'curvature' in feature_names:
            idx = feature_names.index('curvature')
            selected_indices.append(idx)
            selected_features.append('curvature')
        if "Roughness" in feature_set and 'roughness' in feature_names:
            idx = feature_names.index('roughness')
            selected_indices.append(idx)
            selected_features.append('roughness')
        if "Verticality" in feature_set and 'verticality' in feature_names:
            idx = feature_names.index('verticality')
            selected_indices.append(idx)
            selected_features.append('verticality')
        
        # Filter features
        if selected_indices:
            features = features[:, selected_indices]
            feature_names = selected_features
        else:
            logger.warning("No valid features selected, using all features")
    
    logger.info(f"Using features: {feature_names}")
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Generate synthetic labels for demonstration (in a real scenario, this would be training data)
    # For now, we'll use anomaly detection to identify potential defects
    
    if model_type == "Isolation Forest":
        # Use Isolation Forest for unsupervised anomaly detection
        logger.info("Using Isolation Forest for anomaly detection")
        
        model = IsolationForest(contamination=0.05, random_state=42)
        labels = model.fit_predict(scaled_features)
        
        # Convert labels: -1 for anomalies (defects), 1 for normal points
        # Convert to 1 for defects, 0 for normal points
        labels = np.where(labels == -1, 1, 0)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = {
                'features': feature_names,
                'importance': model.feature_importances_
            }
        else:
            feature_importance = None
        
    else:
        # For supervised methods, we need to generate some labeled data 
        # (in a real scenario, this would come from pre-labeled training data)
        
        # Use heuristics to generate "training" data
        # In this case, we'll use roughness and curvature thresholds
        
        # Find roughness and curvature indices if they exist
        roughness_idx = feature_names.index('roughness') if 'roughness' in feature_names else -1
        curvature_idx = feature_names.index('curvature') if 'curvature' in feature_names else -1
        verticality_idx = feature_names.index('verticality') if 'verticality' in feature_names else -1
        
        # Generate synthetic labels based on heuristics
        # High roughness or curvature or very non-vertical surfaces might indicate defects
        synthetic_labels = np.zeros(features.shape[0])
        
        if roughness_idx >= 0:
            roughness = features[:, roughness_idx]
            high_roughness = roughness > np.percentile(roughness, 90)
            synthetic_labels = np.logical_or(synthetic_labels, high_roughness)
            
        if curvature_idx >= 0:
            curvature = features[:, curvature_idx]
            high_curvature = curvature > np.percentile(curvature, 90)
            synthetic_labels = np.logical_or(synthetic_labels, high_curvature)
            
        if verticality_idx >= 0:
            verticality = features[:, verticality_idx]
            # Very non-vertical surfaces on predominantly vertical structures
            abnormal_verticality = np.logical_and(
                verticality < np.percentile(verticality, 10),
                np.median(verticality) > 0.7  # If the structure is mostly vertical
            )
            synthetic_labels = np.logical_or(synthetic_labels, abnormal_verticality)
        
        # Convert to integer labels (0 for normal, 1 for defect)
        synthetic_labels = synthetic_labels.astype(int)
        
        # Split data for training and evaluation
        X_train, X_eval, y_train, y_eval = train_test_split(
            scaled_features, synthetic_labels, test_size=0.3, random_state=42
        )
        
        # Train the selected model
        if model_type == "Random Forest":
            logger.info("Training Random Forest classifier")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Feature importance
            feature_importance = {
                'features': feature_names,
                'importance': model.feature_importances_
            }
            
        elif model_type == "SVM":
            logger.info("Training SVM classifier")
            model = SVC(kernel='rbf', probability=True, random_state=42)
            model.fit(X_train, y_train)
            
            # SVM doesn't provide feature importance
            feature_importance = None
        
        # Evaluate on test set
        y_pred = model.predict(X_eval)
        class_report = sklearn_classification_report(y_eval, y_pred)
        logger.info(f"Evaluation classification report:\n{class_report}")
        
        # Predict on whole dataset
        labels = model.predict(scaled_features)
    
    # Get indices of defect points
    defect_indices = np.where(labels == 1)[0]
    
    # Create classification report
    num_defects = np.sum(labels == 1)
    num_normal = np.sum(labels == 0)
    defect_percentage = (num_defects / len(labels)) * 100
    
    classification_report = {
        'num_points': len(labels),
        'num_defects': int(num_defects),
        'num_normal': int(num_normal),
        'defect_percentage': defect_percentage,
        'model_type': model_type,
        'features_used': feature_names
    }
    
    logger.info(f"Defect detection complete. Found {num_defects} defective points ({defect_percentage:.2f}%)")
    
    return labels, defect_indices, feature_importance, classification_report

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
    logger.info(f"Starting defect detection using {method} clustering")
    
    # Extract points
    points = point_cloud['points']
    
    # Add normals as features if available
    features = points.copy()
    if point_cloud['normals'] is not None:
        normals = point_cloud['normals']
        features = np.hstack((features, normals * 0.5))  # Scale down normals
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply clustering
    if method.lower() == "dbscan":
        logger.info(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = clusterer.fit_predict(scaled_features)
        
        # In DBSCAN, label -1 indicates noise points, which we consider as potential defects
        defect_indices = np.where(labels == -1)[0]
        
        # Convert noise label (-1) to 1 for consistency (defects=1)
        labels = np.where(labels == -1, 1, 0)
        
    else:  # K-Means
        logger.info(f"Running K-Means with {n_clusters} clusters")
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(scaled_features)
        
        # Identify the smallest clusters as potential defects
        # Calculate cluster sizes
        cluster_sizes = np.bincount(cluster_labels)
        
        # Sort clusters by size (ascending)
        sorted_clusters = np.argsort(cluster_sizes)
        
        # Identify small clusters (smallest 20% of clusters or at least 1)
        num_defect_clusters = max(1, int(n_clusters * 0.2))
        defect_clusters = sorted_clusters[:num_defect_clusters]
        
        # Create binary labels (1 for defect, 0 for normal)
        labels = np.zeros_like(cluster_labels)
        for cluster in defect_clusters:
            labels[cluster_labels == cluster] = 1
        
        # Get indices of defect points
        defect_indices = np.where(labels == 1)[0]
    
    num_defects = np.sum(labels == 1)
    defect_percentage = (num_defects / len(labels)) * 100
    
    logger.info(f"Clustering complete. Found {num_defects} potential defect points ({defect_percentage:.2f}%)")
    
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
    from sklearn.cluster import AgglomerativeClustering
    import copy
    from scipy.spatial import KDTree
    
    logger.info(f"Starting defect detection using {method} segmentation")
    
    # Ensure we have normals for the computation
    points = point_cloud['points']
    
    if point_cloud['normals'] is None:
        from preprocessing import preprocess_point_cloud
        # Compute normals using the preprocess function
        temp_cloud = preprocess_point_cloud(
            point_cloud,
            downsample=False,
            remove_outliers=False,
            compute_normals=True
        )
        normals = temp_cloud['normals']
    else:
        normals = point_cloud['normals']
    
    num_points = len(points)
    
    if method == "region_growing":
        logger.info(f"Performing region-based clustering with smoothness={smoothness_threshold}, min_size={min_cluster_size}")
        
        # For region growing, we'll use agglomerative clustering with a custom connectivity matrix
        
        # First, create a KDTree for neighbor queries
        tree = KDTree(points)
        
        # Compute normals similarity matrix (based on dot product)
        similarity_matrix = np.zeros((num_points, num_points))
        connectivity = np.zeros((num_points, num_points), dtype=bool)
        
        # For each point, find its nearest neighbors
        search_radius = 0.1  # Radius for neighbor search
        
        for i in range(num_points):
            # Find neighbors within radius
            indices = tree.query_ball_point(points[i], search_radius)
            
            if len(indices) < 2:  # Not enough neighbors found
                continue
            
            # Compute normal similarity with each neighbor
            for j in indices:
                if i == j:
                    continue
                    
                # Compute angle between normals
                cos_angle = np.dot(normals[i], normals[j])
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angle_degrees = np.degrees(angle)
                
                # If normals are similar (angle small), allow connectivity
                if angle_degrees < smoothness_threshold:
                    connectivity[i, j] = True
                    # Store the similarity (1 is best)
                    similarity_matrix[i, j] = 1.0 - angle_degrees / 180.0
        
        # Perform agglomerative clustering with connectivity constraints
        logger.info("Performing agglomerative clustering...")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - np.cos(np.radians(smoothness_threshold)),
            linkage='average',
            connectivity=connectivity,
        )
        
        try:
            # This can fail if the connectivity matrix doesn't have enough connections
            cluster_labels = clustering.fit_predict(points)
        except Exception as e:
            logger.warning(f"Clustering failed: {e}. Falling back to DBSCAN.")
            # Fall back to DBSCAN
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=search_radius, min_samples=min_cluster_size)
            cluster_labels = clustering.fit_predict(points)
        
        # Count cluster sizes
        unique_clusters, cluster_sizes = np.unique(cluster_labels, return_counts=True)
        
        # Initialize all points as defects (1)
        labels = np.ones(num_points, dtype=int)
        
        # Mark points in large clusters as normal (0)
        for cluster_id, size in zip(unique_clusters, cluster_sizes):
            if size >= min_cluster_size and cluster_id != -1:  # -1 is the noise cluster
                cluster_points = np.where(cluster_labels == cluster_id)[0]
                labels[cluster_points] = 0
    
    else:  # Use RANSAC-like approach for plane detection
        logger.info(f"Performing plane-based segmentation with distance threshold {distance_threshold}")
        
        # Initialize all points as defects (1)
        labels = np.ones(num_points, dtype=int)
        
        # Simple approach to find planar regions using PCA
        # For each point, compute the planarity of its neighborhood
        
        remaining_points = np.arange(num_points)
        segmented_points = set()
        
        # Parameters
        max_iterations = 10  # Maximum number of planes to extract
        search_radius = 0.2  # Radius for local neighborhood
        
        for _ in range(max_iterations):
            if len(remaining_points) < min_cluster_size:
                break
            
            # Choose a random seed point from remaining points
            seed_idx = np.random.choice(remaining_points)
            
            # Find neighbors of seed point
            tree = KDTree(points[remaining_points])
            local_indices = tree.query_ball_point(points[seed_idx], search_radius)
            
            if len(local_indices) < 4:  # Need at least 4 points for plane fitting
                continue
            
            # Get the actual point indices
            local_points = remaining_points[local_indices]
            
            # Compute PCA on local neighborhood
            local_coords = points[local_points]
            centroid = np.mean(local_coords, axis=0)
            local_centered = local_coords - centroid
            
            # Compute covariance matrix
            cov = np.dot(local_centered.T, local_centered) / len(local_points)
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # The smallest eigenvalue corresponds to the normal direction of the plane
            normal = eigenvectors[:, 0]
            
            # Compute distances from all points to this plane
            d = -np.dot(normal, centroid)
            distances = np.abs(np.dot(points, normal) + d)
            
            # Points that are close to the plane are inliers
            inlier_indices = np.where(distances < distance_threshold)[0]
            
            if len(inlier_indices) < min_cluster_size:
                continue
            
            # Mark inliers as normal (0)
            labels[inlier_indices] = 0
            
            # Add inliers to segmented points
            segmented_points.update(inlier_indices)
            
            # Update remaining points
            remaining_points = np.array([i for i in range(num_points) if i not in segmented_points])
            
    # Get indices of defect points
    defect_indices = np.where(labels == 1)[0]
    
    num_defects = len(defect_indices)
    defect_percentage = (num_defects / num_points) * 100
    
    logger.info(f"Segmentation complete. Found {num_defects} potential defect points ({defect_percentage:.2f}%)")
    
    return labels, defect_indices
