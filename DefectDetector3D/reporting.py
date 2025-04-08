import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import json
import base64
from io import BytesIO
import logging
from visualization import visualize_defects

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_report(point_cloud, defect_indices, defect_labels, title="Building Structure Defect Analysis Report", 
                   include_visualizations=True, include_statistics=True, classification_report=None):
    """
    Generate an HTML report for defect analysis.
    
    Args:
        point_cloud (dict): The point cloud dictionary with keys 'points', 'colors', 'normals'
        defect_indices (numpy.ndarray): Indices of points classified as defects
        defect_labels (numpy.ndarray): Labels for all points (1 for defect, 0 for normal)
        title (str): Title of the report
        include_visualizations (bool): Whether to include visualizations
        include_statistics (bool): Whether to include statistics
        classification_report (dict, optional): Classification report data
        
    Returns:
        str: HTML report content
    """
    logger.info("Generating defect analysis report")
    
    # Get current date and time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .container {{
                margin-bottom: 30px;
            }}
            .stat-card {{
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
            }}
            .stat-label {{
                font-size: 14px;
                color: #7f8c8d;
            }}
            .visualization {{
                margin-bottom: 30px;
                height: 500px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 8px 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .footer {{
                margin-top: 50px;
                color: #7f8c8d;
                font-size: 12px;
                text-align: center;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p>Report generated on {now}</p>
    """
    
    # Calculate basic statistics
    points = point_cloud['points']
    num_points = len(points)
    num_defects = len(defect_indices)
    defect_percentage = (num_defects / num_points) * 100 if num_points > 0 else 0
    
    # Add summary section
    html_content += """
        <div class="container">
            <h2>Summary</h2>
            <div class="grid">
    """
    
    # Add summary stats
    html_content += f"""
                <div class="stat-card">
                    <div class="stat-value">{num_points:,}</div>
                    <div class="stat-label">Total Points</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{num_defects:,}</div>
                    <div class="stat-label">Defect Points</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{defect_percentage:.2f}%</div>
                    <div class="stat-label">Defect Percentage</div>
                </div>
    """
    
    # Close summary grid and container
    html_content += """
            </div>
        </div>
    """
    
    # Add visualization if requested
    if include_visualizations:
        logger.info("Adding visualization to report")
        html_content += """
        <div class="container">
            <h2>Defect Visualization</h2>
            <div class="visualization" id="defect-visualization"></div>
        """
        
        # Generate the visualization using Plotly
        fig = visualize_defects(point_cloud, defect_indices, defect_labels)
        
        # Convert to JSON for embedding in HTML
        plot_json = fig.to_json()
        
        # Add JavaScript to render the plot
        html_content += f"""
            <script>
                var plotData = {plot_json};
                Plotly.newPlot('defect-visualization', plotData.data, plotData.layout);
            </script>
        </div>
        """
    
    # Add detailed statistics if requested
    if include_statistics:
        logger.info("Adding detailed statistics to report")
        html_content += """
        <div class="container">
            <h2>Detailed Statistics</h2>
        """
        
        # Coordinate statistics
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        z_coords = points[:, 2]
        
        html_content += """
            <h3>Point Cloud Dimensions</h3>
            <table>
                <tr>
                    <th>Dimension</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Range</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                </tr>
        """
        
        # X dimension
        html_content += f"""
                <tr>
                    <td>X</td>
                    <td>{np.min(x_coords):.4f}</td>
                    <td>{np.max(x_coords):.4f}</td>
                    <td>{np.max(x_coords) - np.min(x_coords):.4f}</td>
                    <td>{np.mean(x_coords):.4f}</td>
                    <td>{np.std(x_coords):.4f}</td>
                </tr>
        """
        
        # Y dimension
        html_content += f"""
                <tr>
                    <td>Y</td>
                    <td>{np.min(y_coords):.4f}</td>
                    <td>{np.max(y_coords):.4f}</td>
                    <td>{np.max(y_coords) - np.min(y_coords):.4f}</td>
                    <td>{np.mean(y_coords):.4f}</td>
                    <td>{np.std(y_coords):.4f}</td>
                </tr>
        """
        
        # Z dimension
        html_content += f"""
                <tr>
                    <td>Z</td>
                    <td>{np.min(z_coords):.4f}</td>
                    <td>{np.max(z_coords):.4f}</td>
                    <td>{np.max(z_coords) - np.min(z_coords):.4f}</td>
                    <td>{np.mean(z_coords):.4f}</td>
                    <td>{np.std(z_coords):.4f}</td>
                </tr>
            </table>
        """
        
        # Add defect distribution by dimension
        if len(defect_indices) > 0:
            html_content += """
            <h3>Defect Distribution</h3>
            <div class="visualization" id="defect-distribution"></div>
            """
            
            # Create a DataFrame for the defect distribution
            df = pd.DataFrame({
                'x': points[:, 0],
                'y': points[:, 1],
                'z': points[:, 2],
                'is_defect': ['Defect' if i in defect_indices else 'Normal' for i in range(len(points))]
            })
            
            # Create histograms for each dimension
            fig_x = px.histogram(df, x='x', color='is_defect', 
                               title='X-Coordinate Distribution',
                               color_discrete_map={'Defect': 'red', 'Normal': 'lightblue'})
            
            fig_y = px.histogram(df, x='y', color='is_defect', 
                               title='Y-Coordinate Distribution',
                               color_discrete_map={'Defect': 'red', 'Normal': 'lightblue'})
            
            fig_z = px.histogram(df, x='z', color='is_defect', 
                               title='Z-Coordinate Distribution',
                               color_discrete_map={'Defect': 'red', 'Normal': 'lightblue'})
            
            # Combine plots into subplots
            from plotly.subplots import make_subplots
            fig_dist = make_subplots(rows=3, cols=1, 
                                  subplot_titles=('X-Coordinate Distribution', 
                                                 'Y-Coordinate Distribution', 
                                                 'Z-Coordinate Distribution'))
            
            # Add traces to subplots
            for trace in fig_x.data:
                fig_dist.add_trace(trace, row=1, col=1)
            
            for trace in fig_y.data:
                fig_dist.add_trace(trace, row=2, col=1)
            
            for trace in fig_z.data:
                fig_dist.add_trace(trace, row=3, col=1)
            
            # Update layout
            fig_dist.update_layout(height=600)
            
            # Convert to JSON for embedding
            plot_json = fig_dist.to_json()
            
            # Add JavaScript to render the plot
            html_content += f"""
                <script>
                    var distData = {plot_json};
                    Plotly.newPlot('defect-distribution', distData.data, distData.layout);
                </script>
            """
        
        # Add classification report if provided
        if classification_report is not None:
            html_content += """
            <h3>Classification Details</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            """
            
            for key, value in classification_report.items():
                if key == 'features_used':
                    features_str = ', '.join(value)
                    html_content += f"""
                        <tr>
                            <td>Features Used</td>
                            <td>{features_str}</td>
                        </tr>
                    """
                elif not isinstance(value, dict):
                    # Format numeric values appropriately
                    if isinstance(value, (int, float)):
                        if key == 'defect_percentage':
                            formatted_value = f"{value:.2f}%"
                        else:
                            formatted_value = f"{value:,}" if isinstance(value, int) else f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    
                    html_content += f"""
                        <tr>
                            <td>{key.replace('_', ' ').title()}</td>
                            <td>{formatted_value}</td>
                        </tr>
                    """
            
            html_content += """
            </table>
            """
        
        # Close the detailed stats container
        html_content += """
        </div>
        """
    
    # Add recommendations section
    html_content += """
    <div class="container">
        <h2>Recommendations</h2>
        <p>Based on the defect detection analysis, the following areas may require attention:</p>
        <ul>
    """
    
    if num_defects > 0:
        # Group defects by spatial proximity
        if len(defect_indices) > 100:
            html_content += f"""
                <li>High concentration of defects detected ({num_defects:,} points, {defect_percentage:.2f}% of structure)</li>
                <li>Recommend detailed inspection of affected areas</li>
            """
            
            # Add more specific recommendations based on defect distribution
            defect_points = points[defect_indices]
            
            # Get bounding box of defects
            min_coords = np.min(defect_points, axis=0)
            max_coords = np.max(defect_points, axis=0)
            
            html_content += f"""
                <li>Defects are concentrated in region bounded by:
                    <ul>
                        <li>X: {min_coords[0]:.2f} to {max_coords[0]:.2f}</li>
                        <li>Y: {min_coords[1]:.2f} to {max_coords[1]:.2f}</li>
                        <li>Z: {min_coords[2]:.2f} to {max_coords[2]:.2f}</li>
                    </ul>
                </li>
            """
        else:
            html_content += f"""
                <li>Isolated defects detected ({num_defects:,} points, {defect_percentage:.2f}% of structure)</li>
                <li>Recommend spot inspection of specific locations</li>
            """
    else:
        html_content += """
            <li>No significant defects detected in the scanned structure</li>
            <li>Recommend regular monitoring to track changes over time</li>
        """
    
    html_content += """
        </ul>
    </div>
    """
    
    # Add footer
    html_content += """
        <div class="footer">
            <p>Generated by Building Structure Defect Detection System</p>
        </div>
    </body>
    </html>
    """
    
    logger.info("Report generation complete")
    return html_content
