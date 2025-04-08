from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                          QGroupBox, QComboBox, QSlider, QCheckBox, QSplitter, QFormLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import os

# Import functionality modules
import utils_pyqt as utils
import visualization_pyqt as visualization


class VisualizationWidget(QWidget):
    """3D visualization widget for BuildScan AI"""
    
    def __init__(self, point_cloud_cache, parent=None):
        super().__init__(parent)
        self.point_cloud_cache = point_cloud_cache
        self.current_point_cloud = None
        self.highlighted_indices = None
        
        # Set up the main layout
        main_layout = QHBoxLayout(self)
        
        # Create the left panel for controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Title
        title_label = QLabel("3D Point Cloud Visualization")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        control_layout.addWidget(title_label)
        
        # Visualization options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout()
        
        # Point size slider
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(10)
        self.point_size_slider.setValue(3)
        self.point_size_slider.setTickPosition(QSlider.TicksBelow)
        self.point_size_slider.setTickInterval(1)
        self.point_size_slider.valueChanged.connect(self._update_point_size)
        viz_layout.addRow("Point Size:", self.point_size_slider)
        
        # Color by feature dropdown
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(["Original Color", "Height", "Distance from Center", "Local Density", "Roughness"])
        self.color_by_combo.currentIndexChanged.connect(self._update_coloring)
        viz_layout.addRow("Color By:", self.color_by_combo)
        
        # Color map dropdown
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "jet", "rainbow"])
        self.colormap_combo.currentIndexChanged.connect(self._update_coloring)
        viz_layout.addRow("Color Map:", self.colormap_combo)
        
        # Show grid checkbox
        self.show_grid_check = QCheckBox("Show Grid")
        self.show_grid_check.setChecked(True)
        self.show_grid_check.stateChanged.connect(self._toggle_grid)
        viz_layout.addRow(self.show_grid_check)
        
        # Show axes checkbox
        self.show_axes_check = QCheckBox("Show Axes")
        self.show_axes_check.setChecked(True)
        self.show_axes_check.stateChanged.connect(self._toggle_axes)
        viz_layout.addRow(self.show_axes_check)
        
        viz_group.setLayout(viz_layout)
        control_layout.addWidget(viz_group)
        
        # View options
        view_group = QGroupBox("View Options")
        view_layout = QVBoxLayout()
        
        # Reset view button
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self._reset_view)
        view_layout.addWidget(self.reset_view_btn)
        
        # Top view button
        self.top_view_btn = QPushButton("Top View")
        self.top_view_btn.clicked.connect(lambda: self._set_view("top"))
        view_layout.addWidget(self.top_view_btn)
        
        # Side view button
        self.side_view_btn = QPushButton("Side View")
        self.side_view_btn.clicked.connect(lambda: self._set_view("side"))
        view_layout.addWidget(self.side_view_btn)
        
        # Front view button
        self.front_view_btn = QPushButton("Front View")
        self.front_view_btn.clicked.connect(lambda: self._set_view("front"))
        view_layout.addWidget(self.front_view_btn)
        
        view_group.setLayout(view_layout)
        control_layout.addWidget(view_group)
        
        # Annotation options
        annotate_group = QGroupBox("Annotations")
        annotate_layout = QVBoxLayout()
        
        # Clear highlights button
        self.clear_highlights_btn = QPushButton("Clear Highlights")
        self.clear_highlights_btn.clicked.connect(self._clear_highlights)
        self.clear_highlights_btn.setEnabled(False)
        annotate_layout.addWidget(self.clear_highlights_btn)
        
        annotate_group.setLayout(annotate_layout)
        control_layout.addWidget(annotate_group)
        
        # Add a spacer at the bottom
        control_layout.addStretch()
        
        # Create the right panel for the 3D viewer
        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        
        # Create the 3D viewer widget
        self.viewer = gl.GLViewWidget()
        self.viewer.setBackgroundColor(pg.mkColor('#f0f0f0'))
        
        # Create grid
        self.grid = gl.GLGridItem()
        self.grid.setSize(10, 10, 1)
        self.grid.setSpacing(1, 1, 1)
        self.viewer.addItem(self.grid)
        
        # Create axes
        self.axes = AxesItem()
        self.viewer.addItem(self.axes)
        
        viewer_layout.addWidget(self.viewer)
        
        # Add status bar for point info
        self.status_label = QLabel("No point cloud loaded")
        viewer_layout.addWidget(self.status_label)
        
        # Create a splitter to allow resizing of panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(viewer_panel)
        splitter.setSizes([200, 800])  # Initial sizes
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Initialize point cloud items
        self.point_cloud_item = None
        self.features_cache = {}
    
    def load_point_cloud(self, point_cloud):
        """Load a point cloud for visualization
        
        Args:
            point_cloud (dict): Point cloud data with 'points' and 'colors' keys
        """
        self.current_point_cloud = point_cloud
        self.highlighted_indices = None
        
        # Clear previous visualization
        if self.point_cloud_item is not None:
            self.viewer.removeItem(self.point_cloud_item)
            self.point_cloud_item = None
        
        # Update status
        if point_cloud is None:
            self.status_label.setText("No point cloud loaded")
            return
        
        # Get point cloud statistics
        stats = utils.get_point_cloud_stats(point_cloud)
        self.status_label.setText(f"Point count: {stats['point_count']:,}")
        
        # Pre-compute features for coloring
        self.features_cache = visualization.compute_features_for_visualization(point_cloud)
        
        # Create visualization
        self._update_visualization()
        
        # Center view on point cloud
        self._reset_view()
        
        # Enable UI elements
        self.clear_highlights_btn.setEnabled(False)
    
    def highlight_defects(self, defect_indices):
        """Highlight defects in the point cloud
        
        Args:
            defect_indices (numpy.ndarray): Indices of points identified as defects
        """
        if self.current_point_cloud is None:
            return
        
        self.highlighted_indices = defect_indices
        
        # Update visualization
        self._update_visualization()
        
        # Update status
        stats = utils.get_point_cloud_stats(self.current_point_cloud)
        if defect_indices is not None and len(defect_indices) > 0:
            defect_percent = (len(defect_indices) / stats['point_count']) * 100
            self.status_label.setText(f"Point count: {stats['point_count']:,} | Defects: {len(defect_indices):,} ({defect_percent:.2f}%)")
            self.clear_highlights_btn.setEnabled(True)
        else:
            self.status_label.setText(f"Point count: {stats['point_count']:,}")
            self.clear_highlights_btn.setEnabled(False)
    
    def _update_visualization(self):
        """Update the point cloud visualization"""
        if self.current_point_cloud is None:
            return
        
        # Clear previous visualization
        if self.point_cloud_item is not None:
            self.viewer.removeItem(self.point_cloud_item)
            self.point_cloud_item = None
        
        # Get visualization data
        viz_data = visualization.visualize_point_cloud(
            self.current_point_cloud, 
            defect_indices=self.highlighted_indices,
            colormap=self.colormap_combo.currentText(),
        )
        
        # Apply custom coloring if selected
        if self.color_by_combo.currentText() != "Original Color" and self.color_by_combo.currentText() in self.features_cache:
            # Get feature values
            feature_values = self.features_cache[self.color_by_combo.currentText()]
            
            # Create color map
            colors = visualization.create_color_map(
                feature_values,
                colormap=self.colormap_combo.currentText()
            )
            
            # If we have highlighted indices, keep them red
            if self.highlighted_indices is not None and len(self.highlighted_indices) > 0:
                colors[self.highlighted_indices] = np.array([1.0, 0.0, 0.0])
            
            # Convert RGB to RGBA (required by GLScatterPlotItem)
            if colors.shape[1] == 3:
                alpha = np.ones((colors.shape[0], 1))
                colors = np.hstack((colors, alpha))
        else:
            colors = viz_data['colors']
            # Convert RGB to RGBA (required by GLScatterPlotItem)
            if colors.shape[1] == 3:
                alpha = np.ones((colors.shape[0], 1))
                colors = np.hstack((colors, alpha))
        
        # Create point cloud item
        self.point_cloud_item = gl.GLScatterPlotItem(
            pos=viz_data['points'],
            color=colors,
            size=self.point_size_slider.value() * 0.1,
            pxMode=True
        )
        self.viewer.addItem(self.point_cloud_item)
    
    def _update_point_size(self):
        """Update the point size in the visualization"""
        if self.point_cloud_item is not None:
            self.point_cloud_item.setSize(self.point_size_slider.value() * 0.1)
    
    def _update_coloring(self):
        """Update the coloring of points"""
        if self.current_point_cloud is not None:
            self._update_visualization()
    
    def _toggle_grid(self, state):
        """Toggle the visibility of the grid"""
        if state == Qt.Checked:
            self.viewer.addItem(self.grid)
        else:
            self.viewer.removeItem(self.grid)
    
    def _toggle_axes(self, state):
        """Toggle the visibility of the axes"""
        if state == Qt.Checked:
            self.viewer.addItem(self.axes)
        else:
            self.viewer.removeItem(self.axes)
    
    def _reset_view(self):
        """Reset the view to center on the point cloud"""
        if self.current_point_cloud is None:
            return
        
        # Calculate center and extent of point cloud
        points = self.current_point_cloud['points']
        center = np.mean(points, axis=0)
        extent = np.max(np.abs(points - center)) * 2
        
        # Set camera position and target
        self.viewer.setCameraPosition(pos=center + np.array([0, -extent, extent/2]), 
                                    distance=extent, 
                                    elevation=30, 
                                    azimuth=45)
    
    def _set_view(self, view_type):
        """Set the view to a specific orientation
        
        Args:
            view_type (str): One of "top", "side", "front"
        """
        if self.current_point_cloud is None:
            return
        
        # Calculate center and extent of point cloud
        points = self.current_point_cloud['points']
        center = np.mean(points, axis=0)
        extent = np.max(np.abs(points - center)) * 2
        
        if view_type == "top":
            # Top view (looking down along Z-axis)
            self.viewer.setCameraPosition(pos=center + np.array([0, 0, extent]), 
                                        distance=extent, 
                                        elevation=90, 
                                        azimuth=0)
        
        elif view_type == "side":
            # Side view (looking along X-axis)
            self.viewer.setCameraPosition(pos=center + np.array([extent, 0, 0]), 
                                        distance=extent, 
                                        elevation=0, 
                                        azimuth=-90)
        
        elif view_type == "front":
            # Front view (looking along Y-axis)
            self.viewer.setCameraPosition(pos=center + np.array([0, extent, 0]), 
                                        distance=extent, 
                                        elevation=0, 
                                        azimuth=0)
    
    def _clear_highlights(self):
        """Clear the highlighted defects"""
        self.highlighted_indices = None
        self._update_visualization()
        
        # Update status
        if self.current_point_cloud:
            stats = utils.get_point_cloud_stats(self.current_point_cloud)
            self.status_label.setText(f"Point count: {stats['point_count']:,}")
        else:
            self.status_label.setText("No point cloud loaded")
        
        self.clear_highlights_btn.setEnabled(False)


class AxesItem(gl.GLAxisItem):
    """Enhanced axis item with labels and colors"""
    
    def __init__(self, size=None):
        super().__init__()
        if size is not None:
            self.setSize(*size)
        else:
            self.setSize(1, 1, 1)
        
        # Set axis colors
        self.x_color = (1, 0, 0, 1)  # Red for X
        self.y_color = (0, 1, 0, 1)  # Green for Y
        self.z_color = (0, 0, 1, 1)  # Blue for Z