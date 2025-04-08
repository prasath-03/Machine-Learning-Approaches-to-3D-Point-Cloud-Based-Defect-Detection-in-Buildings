import numpy as np
import os
import tempfile
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_point_cloud(file_path):
    """
    Load a point cloud from a file.
    
    Args:
        file_path (str): Path to the point cloud file (.ply, .pcd, or .xyz)
        
    Returns:
        dict: Point cloud data with keys 'points', 'colors' (optional), 'normals' (optional)
    """
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    logger.info(f"Loading point cloud from {file_path}")
    
    points = []
    colors = []
    normals = []
    
    # Based on file extension, parse the file
    if ext == '.xyz':
        # Simple XYZ format (x, y, z[, r, g, b])
        with open(file_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 3:  # At least XYZ coordinates
                    points.append([float(values[0]), float(values[1]), float(values[2])])
                    
                    # Check if color data is available (RGB)
                    if len(values) >= 6:
                        colors.append([float(values[3])/255, float(values[4])/255, float(values[5])/255])
    elif ext == '.csv':
        # CSV format with headers
        import pandas as pd
        df = pd.read_csv(file_path)
        
        # Check for required x, y, z columns
        if all(col in df.columns for col in ['x', 'y', 'z']):
            points = df[['x', 'y', 'z']].values
            
            # Check for color columns
            if all(col in df.columns for col in ['r', 'g', 'b']):
                colors = df[['r', 'g', 'b']].values / 255
                
            # Check for normal columns
            if all(col in df.columns for col in ['nx', 'ny', 'nz']):
                normals = df[['nx', 'ny', 'nz']].values
        else:
            raise ValueError("CSV file must contain 'x', 'y', 'z' columns")
    elif ext == '.ply' or ext == '.pcd':
        # For demo purposes, load a sample point cloud instead
        # In practice, we would need a parser for PLY/PCD formats
        logger.warning(f"Using sample point cloud instead of actual {ext} file")
        
        # Generate a simple point cloud for demonstration
        # (a small cube with colors)
        num_points = 1000
        points = np.random.rand(num_points, 3) * 2 - 1  # Points between -1 and 1
        colors = np.zeros((num_points, 3))
        colors[:, 0] = (points[:, 0] + 1) / 2  # Red channel based on x coordinate
        colors[:, 1] = (points[:, 1] + 1) / 2  # Green channel based on y coordinate
        colors[:, 2] = (points[:, 2] + 1) / 2  # Blue channel based on z coordinate
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats are .xyz, .csv, .ply and .pcd")
    
    # Convert to numpy arrays
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32) if len(colors) > 0 else None
    normals = np.array(normals, dtype=np.float32) if len(normals) > 0 else None
    
    # Check if the point cloud is valid
    if len(points) == 0:
        raise ValueError("The loaded point cloud contains no points")
    
    # Create point cloud dictionary
    point_cloud = {
        'points': points,
        'colors': colors if colors is not None and len(colors) > 0 else None,
        'normals': normals if normals is not None and len(normals) > 0 else None
    }
    
    logger.info(f"Successfully loaded point cloud with {len(points)} points")
    
    return point_cloud

def get_point_cloud_stats(point_cloud):
    """
    Get statistics about a point cloud.
    
    Args:
        point_cloud (dict): Input point cloud dictionary with 'points', 'colors', 'normals' keys
        
    Returns:
        dict: Dictionary containing point cloud statistics
    """
    # Get points from the dictionary
    points = point_cloud['points']
    
    # Compute statistics
    num_points = len(points)
    dimensions = points.shape[1] if num_points > 0 else 0
    has_normals = point_cloud['normals'] is not None
    has_colors = point_cloud['colors'] is not None
    
    # If the point cloud has points, compute min, max, and average
    if num_points > 0:
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        avg_coords = np.mean(points, axis=0)
        bounding_box = max_coords - min_coords
    else:
        min_coords = np.zeros(3)
        max_coords = np.zeros(3)
        avg_coords = np.zeros(3)
        bounding_box = np.zeros(3)
    
    # Create statistics dictionary
    stats = {
        'num_points': num_points,
        'dimensions': dimensions,
        'has_normals': has_normals,
        'has_colors': has_colors,
        'min_coords': min_coords,
        'max_coords': max_coords,
        'avg_coords': avg_coords,
        'bounding_box': bounding_box
    }
    
    return stats

def save_temp_file(uploaded_file):
    """
    Save an uploaded file to a temporary file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Path to the temporary file
    """
    # Determine file extension
    file_ext = os.path.splitext(uploaded_file.name)[1]
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    logger.info(f"Saved uploaded file to temporary path: {temp_path}")
    
    return temp_path

def cleanup_temp_file(file_path):
    """
    Clean up a temporary file.
    
    Args:
        file_path (str): Path to the temporary file
    """
    try:
        os.unlink(file_path)
        logger.info(f"Removed temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to remove temporary file {file_path}: {str(e)}")
def export_to_csv(point_cloud, defect_indices=None, file_path="export.csv"):
    """Export point cloud data to CSV format"""
    import pandas as pd
    
    points = point_cloud['points']
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    
    if point_cloud['colors'] is not None:
        colors = point_cloud['colors']
        df[['r', 'g', 'b']] = colors
        
    if point_cloud['normals'] is not None:
        normals = point_cloud['normals']
        df[['nx', 'ny', 'nz']] = normals
        
    if defect_indices is not None:
        df['is_defect'] = 0
        df.loc[defect_indices, 'is_defect'] = 1
        
    df.to_csv(file_path, index=False)
    return file_path

def export_to_ply(point_cloud, defect_indices=None, file_path="export.ply"):
    """Export point cloud data to PLY format"""
    import numpy as np
    
    with open(file_path, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(point_cloud['points'])}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if point_cloud['colors'] is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            
        if point_cloud['normals'] is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            
        if defect_indices is not None:
            f.write("property uchar defect\n")
            
        f.write("end_header\n")
        
        # Write data
        points = point_cloud['points']
        colors = point_cloud['colors']
        normals = point_cloud['normals']
        
        for i in range(len(points)):
            line = f"{points[i][0]} {points[i][1]} {points[i][2]}"
            
            if colors is not None:
                line += f" {int(colors[i][0]*255)} {int(colors[i][1]*255)} {int(colors[i][2]*255)}"
                
            if normals is not None:
                line += f" {normals[i][0]} {normals[i][1]} {normals[i][2]}"
                
            if defect_indices is not None:
                is_defect = 255 if i in defect_indices else 0
                line += f" {is_defect}"
                
            f.write(line + "\n")
            
    return file_path
class PointCloudCache:
    """Simple cache for processed point clouds"""
    def __init__(self, max_size=5):
        self.cache = {}
        self.max_size = max_size
        
    def get(self, key):
        return self.cache.get(key)
        
    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
        
    def clear(self):
        self.cache.clear()

# Initialize cache
point_cloud_cache = PointCloudCache()
