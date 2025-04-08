import numpy as np
import pandas as pd
import os
import tempfile
import csv

class PointCloudCache:
    """Simple cache for processed point clouds"""
    def __init__(self, max_size=5):
        self.cache = {}
        self.max_size = max_size
        
    def get(self, key):
        """Get a point cloud from the cache"""
        return self.cache.get(key)
        
    def put(self, key, value):
        """Add a point cloud to the cache, removing oldest items if needed"""
        if len(self.cache) >= self.max_size:
            # Remove the first item (oldest)
            if self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
        
        self.cache[key] = value
        
    def clear(self):
        """Clear the cache"""
        self.cache.clear()


def load_point_cloud(file_path):
    """
    Load a point cloud from a file.
    
    Args:
        file_path (str): Path to the point cloud file (.ply, .pcd, .xyz, or .csv)
        
    Returns:
        dict: Point cloud data with keys 'points', 'colors' (optional), 'normals' (optional)
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    # Initialize point cloud data structure
    point_cloud = {
        'points': None,
        'colors': None,
        'normals': None
    }
    
    if extension == '.csv':
        # Load from CSV
        points = []
        colors = []
        
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for row in reader:
                # Assume first 3 columns are XYZ coordinates
                x, y, z = map(float, row[:3])
                points.append([x, y, z])
                
                # If there are RGB columns (assume columns 4-6 are RGB)
                if len(row) >= 6:
                    r, g, b = map(float, row[3:6])
                    colors.append([r/255.0, g/255.0, b/255.0])  # Normalize to 0-1
        
        point_cloud['points'] = np.array(points, dtype=np.float32)
        if colors:
            point_cloud['colors'] = np.array(colors, dtype=np.float32)
    
    elif extension == '.xyz':
        # Load from XYZ
        points = []
        colors = []
        
        with open(file_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 3:
                    x, y, z = map(float, values[:3])
                    points.append([x, y, z])
                    
                    # If there are RGB values
                    if len(values) >= 6:
                        r, g, b = map(float, values[3:6])
                        colors.append([r/255.0, g/255.0, b/255.0])  # Normalize to 0-1
        
        point_cloud['points'] = np.array(points, dtype=np.float32)
        if colors:
            point_cloud['colors'] = np.array(colors, dtype=np.float32)
    
    elif extension == '.ply' or extension == '.pcd':
        # For PLY and PCD formats, we'd normally use Open3D, but since we're replacing it:
        # Create a simple parser for basic PLY/PCD files
        if extension == '.ply':
            point_cloud = _parse_ply(file_path)
        else:
            point_cloud = _parse_pcd(file_path)
    
    else:
        raise ValueError(f"Unsupported file format: {extension}")
    
    # Generate random point cloud if file is empty or invalid
    if point_cloud['points'] is None or len(point_cloud['points']) == 0:
        print(f"Warning: {file_path} appears to be empty or invalid. Generating a sample point cloud.")
        point_cloud = _generate_sample_point_cloud()
    
    return point_cloud


def _parse_ply(file_path):
    """
    Parse a PLY file without using Open3D.
    
    Args:
        file_path (str): Path to the PLY file
        
    Returns:
        dict: Point cloud data with keys 'points', 'colors' (optional), 'normals' (optional)
    """
    with open(file_path, 'r') as f:
        # Parse header
        line = f.readline().strip()
        if line != "ply":
            raise ValueError("Not a valid PLY file")
        
        format_line = f.readline().strip()
        if "ascii" not in format_line:
            raise ValueError("Only ASCII PLY format is supported")
        
        # Initialize data
        point_count = 0
        has_colors = False
        has_normals = False
        reading_header = True
        points = []
        colors = []
        normals = []
        
        # Parse header
        while reading_header:
            line = f.readline().strip()
            
            if line.startswith("element vertex"):
                point_count = int(line.split()[-1])
            
            elif line.startswith("property") and "red" in line:
                has_colors = True
            
            elif line.startswith("property") and "nx" in line:
                has_normals = True
            
            elif line == "end_header":
                reading_header = False
        
        # Parse data
        for _ in range(point_count):
            line = f.readline().strip()
            values = line.split()
            
            # First 3 values are always XYZ
            x, y, z = map(float, values[:3])
            points.append([x, y, z])
            
            # Handle normals (typically nx, ny, nz)
            if has_normals:
                nx, ny, nz = map(float, values[3:6])
                normals.append([nx, ny, nz])
                offset = 6
            else:
                offset = 3
            
            # Handle colors (typically red, green, blue)
            if has_colors:
                r, g, b = map(int, values[offset:offset+3])
                colors.append([r/255.0, g/255.0, b/255.0])  # Normalize to 0-1
    
    # Convert to numpy arrays
    point_cloud = {
        'points': np.array(points, dtype=np.float32),
        'colors': np.array(colors, dtype=np.float32) if colors else None,
        'normals': np.array(normals, dtype=np.float32) if normals else None
    }
    
    return point_cloud


def _parse_pcd(file_path):
    """
    Parse a PCD file without using Open3D.
    
    Args:
        file_path (str): Path to the PCD file
        
    Returns:
        dict: Point cloud data with keys 'points', 'colors' (optional), 'normals' (optional)
    """
    with open(file_path, 'r') as f:
        # Parse header
        reading_header = True
        point_count = 0
        has_colors = False
        has_normals = False
        data_type = None
        
        while reading_header:
            line = f.readline().strip()
            
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
                has_colors = 'rgb' in fields or all(c in fields for c in ['r', 'g', 'b'])
                has_normals = all(n in fields for n in ['normal_x', 'normal_y', 'normal_z'])
            
            elif line.startswith("POINTS"):
                point_count = int(line.split()[1])
            
            elif line.startswith("DATA"):
                data_type = line.split()[1]
                reading_header = False
        
        # Only handle ASCII data for simplicity
        if data_type != "ascii":
            raise ValueError("Only ASCII PCD format is supported")
        
        # Parse data
        points = []
        colors = []
        normals = []
        
        for _ in range(point_count):
            line = f.readline().strip()
            values = line.split()
            
            # First 3 values are always XYZ
            x, y, z = map(float, values[:3])
            points.append([x, y, z])
            
            # Handle normals
            if has_normals:
                # Find the indices for normal fields
                nx_idx = fields.index('normal_x')
                ny_idx = fields.index('normal_y')
                nz_idx = fields.index('normal_z')
                nx, ny, nz = map(float, [values[nx_idx], values[ny_idx], values[nz_idx]])
                normals.append([nx, ny, nz])
            
            # Handle colors
            if has_colors:
                if 'rgb' in fields:
                    # RGB packed in a single float (common in PCL)
                    rgb_idx = fields.index('rgb')
                    rgb_val = int(float(values[rgb_idx]))
                    r = (rgb_val >> 16) & 0xFF
                    g = (rgb_val >> 8) & 0xFF
                    b = rgb_val & 0xFF
                    colors.append([r/255.0, g/255.0, b/255.0])
                elif all(c in fields for c in ['r', 'g', 'b']):
                    # RGB as separate fields
                    r_idx = fields.index('r')
                    g_idx = fields.index('g')
                    b_idx = fields.index('b')
                    r, g, b = map(float, [values[r_idx], values[g_idx], values[b_idx]])
                    colors.append([r/255.0, g/255.0, b/255.0])
    
    # Convert to numpy arrays
    point_cloud = {
        'points': np.array(points, dtype=np.float32),
        'colors': np.array(colors, dtype=np.float32) if colors else None,
        'normals': np.array(normals, dtype=np.float32) if normals else None
    }
    
    return point_cloud


def _generate_sample_point_cloud(num_points=1000):
    """
    Generate a sample point cloud for testing.
    
    Args:
        num_points (int): Number of points to generate
        
    Returns:
        dict: Point cloud data with keys 'points', 'colors', 'normals'
    """
    # Generate random points in a sphere
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    r = np.random.uniform(0.8, 1.0, num_points)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    points = np.column_stack((x, y, z))
    
    # Generate random colors
    colors = np.random.uniform(0, 1, (num_points, 3))
    
    # Generate normals (pointing outward from center)
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return {
        'points': points.astype(np.float32),
        'colors': colors.astype(np.float32),
        'normals': normals.astype(np.float32)
    }


def get_point_cloud_stats(point_cloud):
    """
    Get statistics about a point cloud.
    
    Args:
        point_cloud (dict): Input point cloud dictionary with 'points', 'colors', 'normals' keys
        
    Returns:
        dict: Dictionary containing point cloud statistics
    """
    if point_cloud is None or 'points' not in point_cloud or point_cloud['points'] is None:
        return {
            'point_count': 0,
            'dimensions': (0, 0, 0),
            'center': (0, 0, 0),
            'approximate_area': 0
        }
    
    points = point_cloud['points']
    
    # Calculate basic statistics
    point_count = len(points)
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    dimensions = max_bounds - min_bounds
    center = np.mean(points, axis=0)
    
    # Approximate surface area (very rough estimate)
    # For a more accurate estimate, we would need to calculate the actual surface area
    # of the point cloud, which is more complex
    approximate_area = dimensions[0] * dimensions[1] + dimensions[1] * dimensions[2] + dimensions[0] * dimensions[2]
    
    return {
        'point_count': point_count,
        'dimensions': dimensions,
        'center': center,
        'min_bounds': min_bounds,
        'max_bounds': max_bounds,
        'approximate_area': approximate_area
    }


def export_to_csv(point_cloud, defect_indices=None, file_path="export.csv"):
    """Export point cloud data to CSV format"""
    if point_cloud is None or 'points' not in point_cloud or point_cloud['points'] is None:
        return False
    
    points = point_cloud['points']
    colors = point_cloud['colors'] if 'colors' in point_cloud and point_cloud['colors'] is not None else None
    normals = point_cloud['normals'] if 'normals' in point_cloud and point_cloud['normals'] is not None else None
    
    # Create a list for defect labels (1 for defect, 0 for normal)
    defect_labels = np.zeros(len(points), dtype=int)
    if defect_indices is not None:
        defect_labels[defect_indices] = 1
    
    # Create a pandas DataFrame
    data = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2], 'is_defect': defect_labels}
    
    if colors is not None:
        data['red'] = colors[:, 0] * 255
        data['green'] = colors[:, 1] * 255
        data['blue'] = colors[:, 2] * 255
    
    if normals is not None:
        data['nx'] = normals[:, 0]
        data['ny'] = normals[:, 1]
        data['nz'] = normals[:, 2]
    
    df = pd.DataFrame(data)
    
    # Export to CSV
    df.to_csv(file_path, index=False)
    
    return True


def export_to_ply(point_cloud, defect_indices=None, file_path="export.ply"):
    """Export point cloud data to PLY format"""
    if point_cloud is None or 'points' not in point_cloud or point_cloud['points'] is None:
        return False
    
    points = point_cloud['points']
    colors = point_cloud['colors'] if 'colors' in point_cloud and point_cloud['colors'] is not None else None
    normals = point_cloud['normals'] if 'normals' in point_cloud and point_cloud['normals'] is not None else None
    
    # Create a list for defect labels (1 for defect, 0 for normal)
    defect_labels = np.zeros(len(points), dtype=int)
    if defect_indices is not None:
        defect_labels[defect_indices] = 1
    
    # Create defect colors (red for defects)
    if colors is None:
        colors = np.ones((len(points), 3)) * 0.7  # Default gray
    
    # Highlight defects in red
    if defect_indices is not None:
        colors[defect_indices] = np.array([1.0, 0.0, 0.0])  # Red for defects
    
    # Write PLY file
    with open(file_path, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar is_defect\n")
        f.write("end_header\n")
        
        # Write data
        for i in range(len(points)):
            line = f"{points[i, 0]} {points[i, 1]} {points[i, 2]}"
            
            if normals is not None:
                line += f" {normals[i, 0]} {normals[i, 1]} {normals[i, 2]}"
            
            # Convert colors to 0-255 range
            r, g, b = int(colors[i, 0] * 255), int(colors[i, 1] * 255), int(colors[i, 2] * 255)
            line += f" {r} {g} {b} {defect_labels[i]}"
            
            f.write(line + "\n")
    
    return True