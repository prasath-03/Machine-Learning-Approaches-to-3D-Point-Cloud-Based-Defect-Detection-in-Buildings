import numpy as np
import pandas as pd
from scipy.spatial import KDTree

def compare_point_clouds(reference_cloud, comparison_cloud, distance_threshold=0.05):
    """
    Compare two point clouds and identify differences.
    
    Args:
        reference_cloud (dict): Reference point cloud dictionary
        comparison_cloud (dict): Comparison point cloud dictionary
        distance_threshold (float): Distance threshold for considering points as matching
        
    Returns:
        dict: Dictionary with difference metrics and indices
    """
    # Extract points from both clouds
    reference_points = reference_cloud['points']
    comparison_points = comparison_cloud['points']
    
    # Build KD-Tree for efficient nearest neighbor search
    reference_tree = KDTree(reference_points)
    comparison_tree = KDTree(comparison_points)
    
    # Find nearest neighbors and distances (reference to comparison)
    distances_ref_to_comp, _ = reference_tree.query(comparison_points, k=1)
    
    # Find nearest neighbors and distances (comparison to reference)
    distances_comp_to_ref, _ = comparison_tree.query(reference_points, k=1)
    
    # Identify added points (in comparison but not in reference)
    added_indices = np.where(distances_ref_to_comp > distance_threshold)[0]
    
    # Identify removed points (in reference but not in comparison)
    removed_indices = np.where(distances_comp_to_ref > distance_threshold)[0]
    
    # Calculate metrics
    point_count_diff = len(comparison_points) - len(reference_points)
    point_count_diff_percent = (point_count_diff / len(reference_points)) * 100 if len(reference_points) > 0 else 0
    
    # Count matching points
    matching_ref_indices = np.where(distances_comp_to_ref <= distance_threshold)[0]
    matching_comp_indices = np.where(distances_ref_to_comp <= distance_threshold)[0]
    
    # Calculate overlap percentage
    overlap_percent_ref = (len(matching_ref_indices) / len(reference_points)) * 100 if len(reference_points) > 0 else 0
    overlap_percent_comp = (len(matching_comp_indices) / len(comparison_points)) * 100 if len(comparison_points) > 0 else 0
    
    # Calculate average distance between matching points
    avg_distance = np.mean(distances_comp_to_ref[matching_ref_indices]) if len(matching_ref_indices) > 0 else 0
    
    # Calculate volume change
    ref_volume = np.prod(np.max(reference_points, axis=0) - np.min(reference_points, axis=0))
    comp_volume = np.prod(np.max(comparison_points, axis=0) - np.min(comparison_points, axis=0))
    volume_change = comp_volume - ref_volume
    volume_change_percent = (volume_change / ref_volume) * 100 if ref_volume > 0 else 0
    
    # Calculate change severity
    # This is a simple metric based on the percentage of added/removed points
    added_percent = (len(added_indices) / len(comparison_points)) * 100 if len(comparison_points) > 0 else 0
    removed_percent = (len(removed_indices) / len(reference_points)) * 100 if len(reference_points) > 0 else 0
    change_severity = (added_percent + removed_percent) / 2
    
    # Calculate density change
    ref_density = len(reference_points) / ref_volume if ref_volume > 0 else 0
    comp_density = len(comparison_points) / comp_volume if comp_volume > 0 else 0
    density_change = comp_density - ref_density
    density_change_percent = (density_change / ref_density) * 100 if ref_density > 0 else 0
    
    # Return results
    return {
        'added_indices': added_indices,
        'removed_indices': removed_indices,
        'point_count_diff': point_count_diff,
        'point_count_diff_percent': point_count_diff_percent,
        'overlap_percent_ref': overlap_percent_ref,
        'overlap_percent_comp': overlap_percent_comp,
        'avg_distance': avg_distance,
        'volume_change': volume_change,
        'volume_change_percent': volume_change_percent,
        'change_severity': change_severity,
        'density_change': density_change,
        'density_change_percent': density_change_percent
    }


def visualize_point_cloud(point_cloud, defect_indices=None, colormap='viridis', size=1.0):
    """
    Create visualization data for a point cloud.
    
    Args:
        point_cloud (dict): Point cloud dictionary
        defect_indices (numpy.ndarray, optional): Indices of points to highlight as defects
        colormap (str): Name of colormap to use
        size (float): Size of points
        
    Returns:
        dict: Dictionary with visualization data
    """
    # Extract points and colors
    points = point_cloud['points']
    
    # If colors are available, use them, otherwise use default color
    if 'colors' in point_cloud and point_cloud['colors'] is not None:
        colors = point_cloud['colors']
    else:
        # Default color: light blue
        colors = np.tile(np.array([0.5, 0.7, 1.0]), (points.shape[0], 1))
    
    # Highlight defects if indices are provided
    if defect_indices is not None and len(defect_indices) > 0:
        # Make a copy of the colors array to avoid modifying the original
        highlighted_colors = colors.copy()
        # Set color for defects (red)
        highlighted_colors[defect_indices] = np.array([1.0, 0.0, 0.0])
    else:
        highlighted_colors = colors
    
    # Calculate bounds
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    center = np.mean(points, axis=0)
    extent = np.max(np.abs(points - center)) * 2
    
    # Create visualization data dictionary
    viz_data = {
        'points': points,
        'colors': highlighted_colors,
        'min_bounds': min_bounds,
        'max_bounds': max_bounds,
        'center': center,
        'extent': extent
    }
    
    return viz_data


def create_color_map(values, min_val=None, max_val=None, colormap='viridis'):
    """
    Create a color map for values.
    
    Args:
        values (numpy.ndarray): Values to map to colors
        min_val (float, optional): Minimum value for mapping
        max_val (float, optional): Maximum value for mapping
        colormap (str): Name of colormap to use
        
    Returns:
        numpy.ndarray: RGB colors for each value
    """
    import matplotlib.cm as cm
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Determine range
    if min_val is None:
        min_val = np.min(values)
    if max_val is None:
        max_val = np.max(values)
    
    # Normalize values
    if max_val > min_val:
        norm_values = (values - min_val) / (max_val - min_val)
    else:
        norm_values = np.zeros_like(values)
    
    # Clip to [0, 1]
    norm_values = np.clip(norm_values, 0, 1)
    
    # Map to colors
    colors = cmap(norm_values)[:, :3]  # RGB only, no alpha
    
    return colors


def compute_features_for_visualization(point_cloud):
    """
    Compute features for visualization coloring.
    
    Args:
        point_cloud (dict): Point cloud dictionary
        
    Returns:
        dict: Dictionary with features
    """
    points = point_cloud['points']
    normals = point_cloud['normals'] if 'normals' in point_cloud and point_cloud['normals'] is not None else None
    
    # Initialize features
    features = {}
    
    # Height (Z coordinate)
    features['height'] = points[:, 2]
    
    # Distance from center
    center = np.mean(points, axis=0)
    features['distance_from_center'] = np.linalg.norm(points - center, axis=1)
    
    # Local point density (using KNN)
    if len(points) > 10:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=10).fit(points)
        distances, _ = nbrs.kneighbors(points)
        features['local_density'] = np.mean(distances, axis=1)
    
    # Roughness estimation (if normals are available)
    if normals is not None and len(points) > 10:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=10).fit(points)
        _, indices = nbrs.kneighbors(points)
        
        roughness = np.zeros(len(points))
        for i in range(len(points)):
            neighbor_normals = normals[indices[i]]
            # Compute variance of normals as a roughness measure
            normal_var = np.var(neighbor_normals, axis=0)
            roughness[i] = np.sum(normal_var)
        
        features['roughness'] = roughness
    
    return features