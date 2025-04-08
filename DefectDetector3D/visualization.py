import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import logging
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_point_cloud(point_cloud, point_size=2):
    """
    Create an interactive 3D visualization of a point cloud using Plotly.
    
    Args:
        point_cloud (dict): The point cloud dictionary with keys 'points', 'colors', 'normals'
        point_size (int): Size of the points in the visualization
        
    Returns:
        plotly.graph_objects.Figure: Interactive 3D figure
    """
    # Extract points from dictionary
    points = point_cloud['points']
    
    # Initialize color array
    if point_cloud['colors'] is not None:
        # Use existing colors (already in range [0,1])
        colors = point_cloud['colors']
        color_mode = 'rgb'
    else:
        # Use z-coordinate for coloring
        colors = points[:, 2]
        color_mode = 'z'
    
    # Create scatter3d trace
    if color_mode == 'rgb':
        # Convert to RGB format that Plotly understands
        rgb_colors = ['rgb({},{},{})'.format(
            int(r * 255), int(g * 255), int(b * 255)
        ) for r, g, b in colors]
        
        fig = go.Figure(data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=rgb_colors,
                    opacity=0.8
                )
            )
        ])
    else:  # color by z-coordinate
        fig = go.Figure(data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=colors,
                    colorscale='Viridis',
                    opacity=0.8
                )
            )
        ])
    
    # Set layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    
    return fig

def visualize_defects(point_cloud, defect_indices, labels=None, point_size=3):
    """
    Visualize the detected defects in a point cloud.
    
    Args:
        point_cloud (dict): The point cloud dictionary with keys 'points', 'colors', 'normals'
        defect_indices (numpy.ndarray): Indices of points classified as defects
        labels (numpy.ndarray, optional): Labels for all points (1 for defect, 0 for normal)
        point_size (int): Size of the points in the visualization
        
    Returns:
        plotly.graph_objects.Figure: Interactive 3D figure with defects highlighted
    """
    # Check for None values
    if point_cloud is None:
        logger.error("Point cloud is None in visualize_defects")
        # Create a simple empty figure with an error message
        fig = go.Figure()
        fig.update_layout(
            title="Error: No point cloud data available",
            annotations=[dict(
                text="Please upload and process a point cloud first",
                showarrow=False,
                x=0.5,
                y=0.5
            )]
        )
        return fig
        
    if defect_indices is None:
        logger.error("Defect indices is None in visualize_defects")
        # Fall back to regular point cloud visualization
        return visualize_point_cloud(point_cloud, point_size)
    
    try:
        # Extract points
        points = point_cloud['points']
        
        # Create color array
        if labels is None:
            # Create binary labels where defect_indices are 1, others are 0
            labels = np.zeros(len(points), dtype=int)
            labels[defect_indices] = 1
        
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'x': points[:, 0],
            'y': points[:, 1],
            'z': points[:, 2],
            'label': ['Defect' if l == 1 else 'Normal' for l in labels]
        })
        
        # Create 3D scatter plot with categorical colors
        fig = px.scatter_3d(
            df, x='x', y='y', z='z', color='label',
            color_discrete_map={'Defect': 'red', 'Normal': 'lightblue'},
            opacity=0.8
        )
    except Exception as e:
        logger.error(f"Error in visualize_defects: {str(e)}")
        # Create a simple error figure
        fig = go.Figure()
        fig.update_layout(
            title=f"Error visualizing defects: {str(e)}",
            annotations=[dict(
                text="Try running defect detection again",
                showarrow=False,
                x=0.5,
                y=0.5
            )]
        )
        return fig
    
    # Update marker size
    fig.update_traces(marker=dict(size=point_size))
    
    # Set layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        ),
        legend=dict(
            title="Classification",
            x=0.01,
            y=0.99,
            bgcolor="rgba(255, 255, 255, 0.5)"
        )
    )
    
    return fig

def visualize_feature_importance(feature_importance, feature_names):
    """
    Create a bar chart to visualize feature importance.
    
    Args:
        feature_importance (numpy.ndarray): Array of feature importance values
        feature_names (list): List of feature names
        
    Returns:
        plotly.graph_objects.Figure: Bar chart of feature importance
    """
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)
    
    # Create a horizontal bar chart
    fig = go.Figure(go.Bar(
        x=feature_importance[sorted_idx],
        y=[feature_names[i] for i in sorted_idx],
        orientation='h'
    ))
    
    # Set layout
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        margin=dict(l=20, r=20, t=30, b=20),
        height=400
    )
    
    return fig

def visualize_clusters(point_cloud, labels, point_size=3):
    """
    Visualize clustering results.
    
    Args:
        point_cloud (dict): The point cloud dictionary with keys 'points', 'colors', 'normals'
        labels (numpy.ndarray): Cluster labels for each point
        point_size (int): Size of the points in the visualization
        
    Returns:
        plotly.graph_objects.Figure: Interactive 3D figure with clusters colored
    """
    # Extract points
    points = point_cloud['points']
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2],
        'cluster': labels
    })
    
    # Create 3D scatter plot with colors based on cluster
    fig = px.scatter_3d(
        df, x='x', y='y', z='z', color='cluster',
        color_continuous_scale=px.colors.qualitative.G10,
        opacity=0.8
    )
    
    # Update marker size
    fig.update_traces(marker=dict(size=point_size))
    
    # Set layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    
    return fig

def compute_point_cloud_differences(reference_cloud, comparison_cloud, threshold=0.05):
    """
    Compute differences between two point clouds.
    
    Args:
        reference_cloud (dict): Reference point cloud dictionary
        comparison_cloud (dict): Comparison point cloud dictionary 
        threshold (float): Distance threshold for considering points as changed
        
    Returns:
        dict: Dictionary containing difference metrics and classified point indices
    """
    try:
        # Extract points from both clouds
        ref_points = reference_cloud['points']
        comp_points = comparison_cloud['points']
        
        # Build KD-tree for nearest neighbor search
        tree = KDTree(ref_points)
        
        # Find nearest neighbors for each point in comparison cloud
        distances, indices = tree.query(comp_points, k=1)
        
        # Classify points based on distance threshold
        added_mask = distances > threshold  # Points in comparison_cloud not in reference_cloud
        added_indices = np.where(added_mask)[0]
        
        # Find points that were removed (in reference but not in comparison)
        # Build KD-tree for comparison cloud points
        comp_tree = KDTree(comp_points)
        ref_distances, ref_indices = comp_tree.query(ref_points, k=1)
        removed_mask = ref_distances > threshold
        removed_indices = np.where(removed_mask)[0]
        
        # Calculate change statistics
        total_points_ref = len(ref_points)
        total_points_comp = len(comp_points)
        num_added = len(added_indices)
        num_removed = len(removed_indices)
        
        # Points that are present in both clouds but have changed
        unchanged_comp_indices = np.where(~added_mask)[0]
        unchanged_ref_indices = indices[unchanged_comp_indices]
        
        # Calculate distance between matched points to measure change
        changed_distances = distances[~added_mask]
        changed_mask = changed_distances > threshold * 0.5  # Lower threshold for changes
        changed_indices = unchanged_comp_indices[changed_mask]
        
        return {
            'added_indices': added_indices,
            'removed_indices': removed_indices,
            'changed_indices': changed_indices,
            'metrics': {
                'total_points_reference': total_points_ref,
                'total_points_comparison': total_points_comp,
                'num_added': num_added,
                'num_removed': num_removed,
                'num_changed': len(changed_indices),
                'percent_added': (num_added / total_points_comp) * 100 if total_points_comp > 0 else 0,
                'percent_removed': (num_removed / total_points_ref) * 100 if total_points_ref > 0 else 0,
                'percent_changed': (len(changed_indices) / total_points_comp) * 100 if total_points_comp > 0 else 0
            }
        }
    except Exception as e:
        logger.error(f"Error computing point cloud differences: {str(e)}")
        return {
            'added_indices': np.array([]),
            'removed_indices': np.array([]),
            'changed_indices': np.array([]),
            'metrics': {
                'total_points_reference': 0,
                'total_points_comparison': 0,
                'num_added': 0,
                'num_removed': 0,
                'num_changed': 0,
                'percent_added': 0,
                'percent_removed': 0,
                'percent_changed': 0
            },
            'error': str(e)
        }

def visualize_point_cloud_comparison(reference_cloud, comparison_cloud, difference_data=None, point_size=3):
    """
    Visualize two point clouds and highlight the differences between them.
    
    Args:
        reference_cloud (dict): Reference point cloud dictionary
        comparison_cloud (dict): Comparison point cloud dictionary
        difference_data (dict): Dictionary containing difference metrics and point indices
        point_size (int): Size of the points in the visualization
        
    Returns:
        plotly.graph_objects.Figure: Interactive 3D figure with highlighted differences
    """
    if reference_cloud is None or comparison_cloud is None:
        # Create error figure
        fig = go.Figure()
        fig.update_layout(
            title="Error: Missing point cloud data",
            annotations=[dict(
                text="Please upload both reference and comparison point clouds",
                showarrow=False,
                x=0.5,
                y=0.5
            )]
        )
        return fig
    
    try:
        # If difference data not provided, compute it
        if difference_data is None:
            difference_data = compute_point_cloud_differences(reference_cloud, comparison_cloud)
        
        # Extract points from both clouds
        ref_points = reference_cloud['points']
        comp_points = comparison_cloud['points']
        
        # Create dataframes for each point set with category labels
        ref_df = pd.DataFrame({
            'x': ref_points[:, 0],
            'y': ref_points[:, 1],
            'z': ref_points[:, 2],
            'category': 'Reference'
        })
        
        # Add indices of removed points
        if 'removed_indices' in difference_data and len(difference_data['removed_indices']) > 0:
            ref_df.loc[difference_data['removed_indices'], 'category'] = 'Removed'
        
        # Create dataframe for comparison cloud
        comp_df = pd.DataFrame({
            'x': comp_points[:, 0],
            'y': comp_points[:, 1],
            'z': comp_points[:, 2],
            'category': 'Comparison'
        })
        
        # Update categories for added and changed points
        if 'added_indices' in difference_data and len(difference_data['added_indices']) > 0:
            comp_df.loc[difference_data['added_indices'], 'category'] = 'Added'
        
        if 'changed_indices' in difference_data and len(difference_data['changed_indices']) > 0:
            comp_df.loc[difference_data['changed_indices'], 'category'] = 'Changed'
        
        # Combine dataframes for visualization
        combined_df = pd.concat([ref_df, comp_df], ignore_index=True)
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            combined_df, x='x', y='y', z='z', color='category',
            color_discrete_map={
                'Reference': 'lightgray',
                'Comparison': 'lightblue',
                'Added': 'green',
                'Removed': 'red',
                'Changed': 'orange'
            },
            opacity=0.7
        )
        
        # Update marker size
        fig.update_traces(marker=dict(size=point_size))
        
        # Set layout
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z')
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            legend=dict(
                title="Point Category",
                x=0.01,
                y=0.99,
                bgcolor="rgba(255, 255, 255, 0.5)"
            )
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error in visualize_point_cloud_comparison: {str(e)}")
        # Create a simple error figure
        fig = go.Figure()
        fig.update_layout(
            title=f"Error visualizing point cloud comparison: {str(e)}",
            annotations=[dict(
                text="An error occurred during visualization",
                showarrow=False,
                x=0.5,
                y=0.5
            )]
        )
        return fig

def visualize_comparison_metrics(metrics, view_type='bar'):
    """
    Visualize point cloud comparison metrics.
    
    Args:
        metrics (dict): Dictionary of comparison metrics
        view_type (str): Visualization type ('bar' or 'pie')
        
    Returns:
        plotly.graph_objects.Figure: Chart showing comparison metrics
    """
    if metrics is None:
        fig = go.Figure()
        fig.update_layout(
            title="Error: No comparison metrics available",
            annotations=[dict(
                text="Please run point cloud comparison first",
                showarrow=False,
                x=0.5,
                y=0.5
            )]
        )
        return fig
    
    try:
        # Extract key metrics for visualization
        labels = ['Added', 'Removed', 'Changed', 'Unchanged']
        
        # Calculate unchanged percentage
        unchanged_percent = 100 - (metrics['percent_added'] + metrics['percent_removed'] + metrics['percent_changed'])
        if unchanged_percent < 0:
            unchanged_percent = 0
            
        values = [
            metrics['percent_added'],
            metrics['percent_removed'],
            metrics['percent_changed'],
            unchanged_percent
        ]
        
        # Absolute counts for text display
        absolute_counts = [
            metrics['num_added'],
            metrics['num_removed'],
            metrics['num_changed'],
            metrics['total_points_comparison'] - metrics['num_added'] - metrics['num_changed']
        ]
        
        if view_type == 'pie':
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                textinfo='percent+label',
                hoverinfo='label+percent+text',
                hovertext=[f"{count} points" for count in absolute_counts],
                marker_colors=['green', 'red', 'orange', 'lightgray']
            )])
            
            fig.update_layout(
                title="Point Cloud Comparison: Percentage of Points by Category",
                margin=dict(l=20, r=20, t=50, b=20),
                height=400
            )
        else:  # bar chart
            # Create horizontal bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=values,
                    y=labels,
                    orientation='h',
                    text=[f"{v:.1f}% ({absolute_counts[i]} points)" for i, v in enumerate(values)],
                    textposition='auto',
                    marker_color=['green', 'red', 'orange', 'lightgray']
                )
            ])
            
            fig.update_layout(
                title="Point Cloud Comparison: Percentage of Points by Category",
                xaxis_title="Percentage",
                yaxis_title="Category",
                margin=dict(l=20, r=20, t=50, b=20),
                height=400
            )
            
        return fig
    except Exception as e:
        logger.error(f"Error in visualize_comparison_metrics: {str(e)}")
        # Create error figure
        fig = go.Figure()
        fig.update_layout(
            title=f"Error visualizing comparison metrics: {str(e)}",
            annotations=[dict(
                text="An error occurred during visualization",
                showarrow=False,
                x=0.5,
                y=0.5
            )]
        )
        return fig
