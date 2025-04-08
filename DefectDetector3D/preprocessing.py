import numpy as np
import copy
import logging
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_point_cloud(point_cloud, 
                          downsample=True, 
                          voxel_size=0.05, 
                          remove_outliers=True, 
                          nb_neighbors=20, 
                          std_ratio=2.0,
                          compute_normals=True,
                          search_radius=0.1,
                          progress_callback=None):
    """
    Preprocess a point cloud by applying common operations like downsampling,
    outlier removal, and normal computation.
    
    Args:
        point_cloud (dict): Input point cloud dictionary with 'points', 'colors', 'normals' keys
        downsample (bool): Whether to downsample the point cloud
        voxel_size (float): Voxel size for downsampling
        remove_outliers (bool): Whether to remove outliers
        nb_neighbors (int): Number of neighbors for outlier removal
        std_ratio (float): Standard deviation ratio for outlier removal
        compute_normals (bool): Whether to compute normals
        search_radius (float): Search radius for normal estimation
        
    Returns:
        dict: Preprocessed point cloud dictionary
    """
    # Make a copy to avoid modifying the original
    processed_cloud = copy.deepcopy(point_cloud)
    points = processed_cloud['points']
    
    # Get initial point count
    initial_point_count = len(points)
    logger.info(f"Initial point count: {initial_point_count}")
    
    # Downsample using voxel grid
    if downsample and voxel_size > 0:
        logger.info(f"Downsampling with voxel size: {voxel_size}")
        
        # Simple voxel downsampling implementation
        voxel_grid = {}
        for i, point in enumerate(points):
            # Calculate voxel index
            voxel_idx = tuple((point // voxel_size).astype(int))
            
            # Store point index in voxel
            if voxel_idx not in voxel_grid:
                voxel_grid[voxel_idx] = [i]
            else:
                voxel_grid[voxel_idx].append(i)
        
        # Select one point per voxel (the centroid)
        selected_indices = []
        for voxel_indices in voxel_grid.values():
            if len(voxel_indices) > 0:
                # Calculate centroid of points in this voxel
                voxel_points = points[voxel_indices]
                centroid_idx = voxel_indices[np.argmin(np.sum((voxel_points - np.mean(voxel_points, axis=0))**2, axis=1))]
                selected_indices.append(centroid_idx)
        
        # Update point cloud with selected points
        processed_cloud['points'] = points[selected_indices]
        
        # Also filter colors if they exist
        if processed_cloud['colors'] is not None:
            processed_cloud['colors'] = processed_cloud['colors'][selected_indices]
        
        # Reset normals as they need to be recomputed
        processed_cloud['normals'] = None
        
        logger.info(f"After downsampling: {len(processed_cloud['points'])} points")
    
    # Remove statistical outliers
    if remove_outliers and nb_neighbors > 0:
        logger.info(f"Removing outliers with {nb_neighbors} neighbors and std ratio {std_ratio}")
        
        # Get updated points
        points = processed_cloud['points']
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=nb_neighbors, algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Calculate mean distance for each point to its neighbors
        mean_distances = np.mean(distances[:, 1:], axis=1)  # Skip first neighbor (self)
        
        # Calculate mean and standard deviation of all mean distances
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        
        # Define threshold based on std_ratio
        threshold = global_mean + std_ratio * global_std
        
        # Find indices of inliers
        inlier_indices = np.where(mean_distances < threshold)[0]
        
        # Update point cloud with inliers
        processed_cloud['points'] = points[inlier_indices]
        
        # Also filter colors if they exist
        if processed_cloud['colors'] is not None:
            processed_cloud['colors'] = processed_cloud['colors'][inlier_indices]
            
        # Also filter normals if they exist
        if processed_cloud['normals'] is not None:
            processed_cloud['normals'] = processed_cloud['normals'][inlier_indices]
        
        logger.info(f"After outlier removal: {len(processed_cloud['points'])} points")
    
    # Compute normals
    if compute_normals:
        logger.info(f"Computing normals with search radius: {search_radius}")
        
        # Get updated points
        points = processed_cloud['points']
        normals = np.zeros_like(points)
        
        # Create KDTree for neighbor searches
        tree = KDTree(points)
        
        # For each point, find neighbors and compute normal
        for i, point in enumerate(points):
            # Find neighbors within search radius
            indices = tree.query_ball_point(point, search_radius)
            
            if len(indices) < 3:  # Need at least 3 points for PCA
                # If not enough neighbors, increase search radius
                indices = tree.query(point, k=min(30, len(points)))[1]
            
            if len(indices) < 3:  # Still not enough, set a default normal
                normals[i] = [0, 0, 1]
                continue
                
            # Get neighboring points
            neighbors = points[indices]
            
            # Calculate centroid
            centroid = np.mean(neighbors, axis=0)
            
            # Form covariance matrix
            cov = np.zeros((3, 3))
            for neighbor in neighbors:
                diff = neighbor - centroid
                cov += np.outer(diff, diff)
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Normal is eigenvector with smallest eigenvalue
            normal = eigenvectors[:, 0]
            
            # Ensure normal points outward from centroid
            if np.dot(normal, point - centroid) < 0:
                normal = -normal
                
            normals[i] = normal / np.linalg.norm(normal)  # Normalize
        
        # Store computed normals
        processed_cloud['normals'] = normals
        logger.info("Normals computed")
    
    # Final point count
    final_point_count = len(processed_cloud['points'])
    logger.info(f"Final point count: {final_point_count}")
    logger.info(f"Points removed: {initial_point_count - final_point_count} ({(initial_point_count - final_point_count) / initial_point_count * 100:.2f}%)")
    
    return processed_cloud

def extract_features(point_cloud):
    """
    Extract geometric features from a point cloud for defect detection.
    
    Args:
        point_cloud (dict): Input point cloud dictionary with 'points', 'colors', 'normals' keys
        
    Returns:
        numpy.ndarray: Feature matrix
        list: Feature names
    """
    # Get points and normals
    points = point_cloud['points']
    
    # Check if normals are available
    if point_cloud['normals'] is None:
        logger.warning("Point cloud doesn't have normals. Computing them...")
        # Compute normals using the preprocess function with only normals=True
        temp_cloud = preprocess_point_cloud(
            point_cloud,
            downsample=False,
            remove_outliers=False,
            compute_normals=True
        )
        normals = temp_cloud['normals']
    else:
        normals = point_cloud['normals']
    
    # Initialize feature matrix
    features = []
    feature_names = []
    
    # 1. Add normal vector components
    features.append(normals)
    feature_names.extend(['normal_x', 'normal_y', 'normal_z'])
    
    # 2. Calculate verticality (angle with up vector)
    up_vector = np.array([0, 0, 1])  # Assuming Z is up
    verticality = np.abs(np.dot(normals, up_vector))
    features.append(verticality.reshape(-1, 1))
    feature_names.append('verticality')
    
    # 3. Compute local curvature (using PCA on local neighborhoods)
    tree = KDTree(points)
    curvature = np.zeros(len(points))
    
    for i in range(len(points)):
        # Find neighboring points (30 nearest neighbors)
        indices = tree.query(points[i], k=min(30, len(points)))[1]
        
        if len(indices) < 3:  # Need at least 3 points for PCA
            continue
            
        # Get neighboring points
        neighbors = points[indices]
        
        # Compute PCA
        cov = np.cov(neighbors.T)
        eigenvalues, _ = np.linalg.eigh(cov)
        
        # Sort eigenvalues in ascending order
        eigenvalues = np.sort(eigenvalues)
        
        # Compute curvature as smallest eigenvalue / sum of eigenvalues
        if np.sum(eigenvalues) > 0:
            curvature[i] = eigenvalues[0] / np.sum(eigenvalues)
    
    features.append(curvature.reshape(-1, 1))
    feature_names.append('curvature')
    
    # 4. Compute roughness (variation of normals in local neighborhood)
    roughness = np.zeros(len(points))
    
    for i in range(len(points)):
        # Find neighboring points (30 nearest neighbors)
        indices = tree.query(points[i], k=min(30, len(points)))[1]
        
        if len(indices) < 2:
            continue
            
        # Get neighboring normals
        neighbor_normals = normals[indices]
        
        # Compute variance of normals
        mean_normal = np.mean(neighbor_normals, axis=0)
        if np.linalg.norm(mean_normal) > 0:
            mean_normal = mean_normal / np.linalg.norm(mean_normal)  # Normalize
        
        # Sum of squared distances from mean normal
        variances = np.sum((neighbor_normals - mean_normal)**2, axis=1)
        roughness[i] = np.mean(variances)
    
    features.append(roughness.reshape(-1, 1))
    feature_names.append('roughness')
    
    # Combine all features
    feature_matrix = np.hstack(features)
    
    return feature_matrix, feature_names
def batch_process_point_clouds(point_clouds, process_config):
    """Process multiple point clouds with the same configuration"""
    processed_clouds = []
    for cloud in point_clouds:
        processed_cloud = preprocess_point_cloud(
            cloud,
            downsample=process_config.get('downsample', True),
            voxel_size=process_config.get('voxel_size', 0.05),
            remove_outliers=process_config.get('remove_outliers', True),
            nb_neighbors=process_config.get('nb_neighbors', 20),
            std_ratio=process_config.get('std_ratio', 2.0),
            compute_normals=process_config.get('compute_normals', True),
            search_radius=process_config.get('search_radius', 0.1)
        )
        processed_clouds.append(processed_cloud)
    return processed_clouds
