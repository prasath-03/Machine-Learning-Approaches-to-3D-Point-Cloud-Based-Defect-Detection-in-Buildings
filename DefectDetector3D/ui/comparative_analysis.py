from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                          QGroupBox, QFileDialog, QComboBox, QSlider, QDoubleSpinBox,
                          QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                          QSplitter, QFormLayout, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import os

# Import core functionality modules
import utils_pyqt as utils
import visualization_pyqt as visualization


class DualPointCloudViewer(QWidget):
    """Widget for displaying two point clouds side by side for comparison"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up the layout
        layout = QHBoxLayout(self)
        
        # Create splitter for two viewers
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Create the reference cloud viewer
        self.reference_frame = QWidget()
        reference_layout = QVBoxLayout(self.reference_frame)
        reference_label = QLabel("Reference Point Cloud")
        reference_label.setAlignment(Qt.AlignCenter)
        reference_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        
        self.reference_viewer = gl.GLViewWidget()
        self.reference_viewer.setBackgroundColor(pg.mkColor('#f0f0f0'))
        self.reference_viewer.addItem(gl.GLGridItem())
        
        reference_layout.addWidget(reference_label)
        reference_layout.addWidget(self.reference_viewer)
        
        # Create the comparison cloud viewer
        self.comparison_frame = QWidget()
        comparison_layout = QVBoxLayout(self.comparison_frame)
        comparison_label = QLabel("Comparison Point Cloud")
        comparison_label.setAlignment(Qt.AlignCenter)
        comparison_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        
        self.comparison_viewer = gl.GLViewWidget()
        self.comparison_viewer.setBackgroundColor(pg.mkColor('#f0f0f0'))
        self.comparison_viewer.addItem(gl.GLGridItem())
        
        comparison_layout.addWidget(comparison_label)
        comparison_layout.addWidget(self.comparison_viewer)
        
        # Add frames to splitter
        self.splitter.addWidget(self.reference_frame)
        self.splitter.addWidget(self.comparison_frame)
        
        # Add splitter to main layout
        layout.addWidget(self.splitter)
        
        # Initialize point cloud items
        self.reference_point_cloud_item = None
        self.comparison_point_cloud_item = None
        self.difference_points_item_ref = None
        self.difference_points_item_comp = None
    
    def display_reference_cloud(self, point_cloud):
        """Display reference point cloud
        
        Args:
            point_cloud (dict): Point cloud data with 'points' and 'colors' keys
        """
        # Clear previous visualization
        if self.reference_point_cloud_item is not None:
            self.reference_viewer.removeItem(self.reference_point_cloud_item)
            self.reference_point_cloud_item = None
        
        if self.difference_points_item_ref is not None:
            self.reference_viewer.removeItem(self.difference_points_item_ref)
            self.difference_points_item_ref = None
        
        points = point_cloud['points']
        
        # If colors are available, use them, otherwise use default color
        if 'colors' in point_cloud and point_cloud['colors'] is not None:
            colors = point_cloud['colors']
            # Convert RGB to RGBA (required by GLScatterPlotItem)
            if colors.shape[1] == 3:
                alpha = np.ones((colors.shape[0], 1))
                colors = np.hstack((colors, alpha))
        else:
            # Default color: light blue
            colors = np.tile(np.array([0.5, 0.7, 1.0, 1.0]), (points.shape[0], 1))
        
        # Create and add point cloud item
        self.reference_point_cloud_item = gl.GLScatterPlotItem(
            pos=points, 
            color=colors,
            size=0.5,
            pxMode=True
        )
        self.reference_viewer.addItem(self.reference_point_cloud_item)
        
        # Center the view on the point cloud
        self._center_on_point_cloud(self.reference_viewer, points)
    
    def display_comparison_cloud(self, point_cloud):
        """Display comparison point cloud
        
        Args:
            point_cloud (dict): Point cloud data with 'points' and 'colors' keys
        """
        # Clear previous visualization
        if self.comparison_point_cloud_item is not None:
            self.comparison_viewer.removeItem(self.comparison_point_cloud_item)
            self.comparison_point_cloud_item = None
        
        if self.difference_points_item_comp is not None:
            self.comparison_viewer.removeItem(self.difference_points_item_comp)
            self.difference_points_item_comp = None
        
        points = point_cloud['points']
        
        # If colors are available, use them, otherwise use default color
        if 'colors' in point_cloud and point_cloud['colors'] is not None:
            colors = point_cloud['colors']
            # Convert RGB to RGBA (required by GLScatterPlotItem)
            if colors.shape[1] == 3:
                alpha = np.ones((colors.shape[0], 1))
                colors = np.hstack((colors, alpha))
        else:
            # Default color: light green
            colors = np.tile(np.array([0.5, 1.0, 0.7, 1.0]), (points.shape[0], 1))
        
        # Create and add point cloud item
        self.comparison_point_cloud_item = gl.GLScatterPlotItem(
            pos=points, 
            color=colors,
            size=0.5,
            pxMode=True
        )
        self.comparison_viewer.addItem(self.comparison_point_cloud_item)
        
        # Center the view on the point cloud
        self._center_on_point_cloud(self.comparison_viewer, points)
    
    def highlight_differences(self, reference_cloud, comparison_cloud, difference_data):
        """Highlight the differences between two point clouds
        
        Args:
            reference_cloud (dict): Reference point cloud data
            comparison_cloud (dict): Comparison point cloud data
            difference_data (dict): Dictionary with difference information
        """
        # Clear previous difference highlights
        if self.difference_points_item_ref is not None:
            self.reference_viewer.removeItem(self.difference_points_item_ref)
            self.difference_points_item_ref = None
        
        if self.difference_points_item_comp is not None:
            self.comparison_viewer.removeItem(self.difference_points_item_comp)
            self.difference_points_item_comp = None
        
        if difference_data and 'removed_indices' in difference_data and 'added_indices' in difference_data:
            # Highlight removed points in reference cloud (points that don't exist in comparison)
            if len(difference_data['removed_indices']) > 0:
                removed_points = reference_cloud['points'][difference_data['removed_indices']]
                removed_colors = np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (len(difference_data['removed_indices']), 1))
                
                self.difference_points_item_ref = gl.GLScatterPlotItem(
                    pos=removed_points,
                    color=removed_colors,
                    size=2.0,
                    pxMode=True
                )
                self.reference_viewer.addItem(self.difference_points_item_ref)
            
            # Highlight added points in comparison cloud (points that don't exist in reference)
            if len(difference_data['added_indices']) > 0:
                added_points = comparison_cloud['points'][difference_data['added_indices']]
                added_colors = np.tile(np.array([0.0, 1.0, 0.0, 1.0]), (len(difference_data['added_indices']), 1))
                
                self.difference_points_item_comp = gl.GLScatterPlotItem(
                    pos=added_points,
                    color=added_colors,
                    size=2.0,
                    pxMode=True
                )
                self.comparison_viewer.addItem(self.difference_points_item_comp)
    
    def _center_on_point_cloud(self, viewer, points):
        """Center viewer on the point cloud"""
        # Calculate center and extent of point cloud
        center = np.mean(points, axis=0)
        extent = np.max(np.abs(points - center)) * 2
        
        # Set camera position and target
        viewer.setCameraPosition(pos=center + np.array([0, -extent, extent/2]), 
                               distance=extent,
                               elevation=30, 
                               azimuth=45)


class ComparativeAnalysisWidget(QWidget):
    """Comparative analysis widget for BuildScan AI"""
    
    def __init__(self, point_cloud_cache, parent=None):
        super().__init__(parent)
        self.point_cloud_cache = point_cloud_cache
        self.reference_point_cloud = None
        self.comparison_point_cloud = None
        self.difference_data = None
        
        # Set up the main layout
        main_layout = QVBoxLayout(self)
        
        # Create header section
        header_layout = QHBoxLayout()
        title_label = QLabel("Comparative Analysis")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Create file selection section
        file_selection_layout = QHBoxLayout()
        
        # Reference point cloud selection
        reference_group = QGroupBox("Reference Point Cloud")
        reference_layout = QVBoxLayout()
        self.reference_combo = QComboBox()
        self.reference_combo.setEditable(False)
        self.reference_browse_btn = QPushButton("Browse...")
        self.reference_browse_btn.clicked.connect(lambda: self._browse_point_cloud("reference"))
        
        reference_layout.addWidget(self.reference_combo)
        reference_layout.addWidget(self.reference_browse_btn)
        reference_group.setLayout(reference_layout)
        
        # Comparison point cloud selection
        comparison_group = QGroupBox("Comparison Point Cloud")
        comparison_layout = QVBoxLayout()
        self.comparison_combo = QComboBox()
        self.comparison_combo.setEditable(False)
        self.comparison_browse_btn = QPushButton("Browse...")
        self.comparison_browse_btn.clicked.connect(lambda: self._browse_point_cloud("comparison"))
        
        comparison_layout.addWidget(self.comparison_combo)
        comparison_layout.addWidget(self.comparison_browse_btn)
        comparison_group.setLayout(comparison_layout)
        
        file_selection_layout.addWidget(reference_group)
        file_selection_layout.addWidget(comparison_group)
        
        # Comparison settings
        settings_group = QGroupBox("Comparison Settings")
        settings_layout = QFormLayout()
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setMinimum(0.001)
        self.threshold_spin.setMaximum(1.0)
        self.threshold_spin.setSingleStep(0.001)
        self.threshold_spin.setValue(0.05)
        settings_layout.addRow("Distance Threshold (m):", self.threshold_spin)
        
        self.run_comparison_btn = QPushButton("Compare Point Clouds")
        self.run_comparison_btn.clicked.connect(self._run_comparison)
        self.run_comparison_btn.setEnabled(False)
        
        settings_layout.addRow(self.run_comparison_btn)
        settings_group.setLayout(settings_layout)
        
        # Add dual viewer
        self.dual_viewer = DualPointCloudViewer()
        
        # Results table
        results_group = QGroupBox("Comparison Results")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels([
            "Metric", "Value", "Description", "Impact"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)
        
        # Add all sections to the main layout
        main_layout.addLayout(header_layout)
        main_layout.addLayout(file_selection_layout)
        main_layout.addWidget(settings_group)
        main_layout.addWidget(self.dual_viewer, stretch=3)
        main_layout.addWidget(results_group, stretch=1)
        
        # Initialize UI
        self._populate_cloud_combos()
        
        # Connect signals
        self.reference_combo.currentIndexChanged.connect(self._handle_reference_changed)
        self.comparison_combo.currentIndexChanged.connect(self._handle_comparison_changed)
    
    def _populate_cloud_combos(self):
        """Populate the point cloud selection combos with available files"""
        # Clear combos
        self.reference_combo.clear()
        self.comparison_combo.clear()
        
        # Add cache entries
        for key in self.point_cloud_cache.cache:
            self.reference_combo.addItem(key)
            self.comparison_combo.addItem(key)
        
        # Enable/disable comparison button
        self._update_comparison_button()
    
    def _browse_point_cloud(self, target):
        """Open a file dialog to browse for a point cloud
        
        Args:
            target (str): Either 'reference' or 'comparison'
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Open {target.capitalize()} Point Cloud File",
            "",
            "Point Cloud Files (*.ply *.pcd *.xyz *.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load the point cloud
                point_cloud = utils.load_point_cloud(file_path)
                
                # Update cache
                file_name = os.path.basename(file_path)
                self.point_cloud_cache.put(file_name, point_cloud)
                
                # Update combos
                self._populate_cloud_combos()
                
                # Select the new file
                idx = self.reference_combo.findText(file_name) if target == 'reference' else self.comparison_combo.findText(file_name)
                if idx >= 0:
                    if target == 'reference':
                        self.reference_combo.setCurrentIndex(idx)
                    else:
                        self.comparison_combo.setCurrentIndex(idx)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load point cloud: {str(e)}")
    
    def _handle_reference_changed(self, index):
        """Handle selection change in reference combo"""
        if index >= 0:
            reference_name = self.reference_combo.currentText()
            self.reference_point_cloud = self.point_cloud_cache.get(reference_name)
            
            if self.reference_point_cloud:
                self.dual_viewer.display_reference_cloud(self.reference_point_cloud)
            
            self._update_comparison_button()
    
    def _handle_comparison_changed(self, index):
        """Handle selection change in comparison combo"""
        if index >= 0:
            comparison_name = self.comparison_combo.currentText()
            self.comparison_point_cloud = self.point_cloud_cache.get(comparison_name)
            
            if self.comparison_point_cloud:
                self.dual_viewer.display_comparison_cloud(self.comparison_point_cloud)
            
            self._update_comparison_button()
    
    def _update_comparison_button(self):
        """Update the state of the comparison button"""
        enable = (self.reference_point_cloud is not None and 
                 self.comparison_point_cloud is not None and
                 self.reference_combo.currentText() != self.comparison_combo.currentText())
        
        self.run_comparison_btn.setEnabled(enable)
    
    def _run_comparison(self):
        """Run the comparison between reference and comparison point clouds"""
        if not self.reference_point_cloud or not self.comparison_point_cloud:
            return
        
        threshold = self.threshold_spin.value()
        
        try:
            # Run comparison
            self.difference_data = visualization.compare_point_clouds(
                self.reference_point_cloud,
                self.comparison_point_cloud,
                distance_threshold=threshold
            )
            
            # Highlight differences
            self.dual_viewer.highlight_differences(
                self.reference_point_cloud,
                self.comparison_point_cloud,
                self.difference_data
            )
            
            # Update results table
            self._update_results_table(self.difference_data)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compare point clouds: {str(e)}")
    
    def _update_results_table(self, difference_data):
        """Update the results table with comparison metrics
        
        Args:
            difference_data (dict): Dictionary with difference metrics
        """
        # Clear the table
        self.results_table.setRowCount(0)
        
        if not difference_data:
            return
        
        # Metrics to display in the table
        metrics = [
            {
                'name': 'Added Points',
                'value': f"{len(difference_data['added_indices']):,}",
                'description': 'Points present in comparison but not in reference',
                'impact': 'New structure or additions'
            },
            {
                'name': 'Removed Points',
                'value': f"{len(difference_data['removed_indices']):,}",
                'description': 'Points present in reference but not in comparison',
                'impact': 'Missing structure or removed elements'
            },
            {
                'name': 'Point Count Difference',
                'value': f"{difference_data['point_count_diff']:,} ({difference_data['point_count_diff_percent']:.2f}%)",
                'description': 'Difference in total point count',
                'impact': 'Overall scan density change'
            },
            {
                'name': 'Reference Overlap',
                'value': f"{difference_data['overlap_percent_ref']:.2f}%",
                'description': 'Percentage of reference points with matches',
                'impact': 'Coverage consistency'
            },
            {
                'name': 'Comparison Overlap',
                'value': f"{difference_data['overlap_percent_comp']:.2f}%",
                'description': 'Percentage of comparison points with matches',
                'impact': 'Scan coverage'
            },
            {
                'name': 'Average Distance',
                'value': f"{difference_data['avg_distance']:.4f} m",
                'description': 'Average distance between matching points',
                'impact': 'Alignment precision'
            },
            {
                'name': 'Volume Change',
                'value': f"{difference_data['volume_change']:.2f} m³ ({difference_data['volume_change_percent']:.2f}%)",
                'description': 'Change in volume of point cloud',
                'impact': 'Structural expansion or contraction'
            },
            {
                'name': 'Density Change',
                'value': f"{difference_data['density_change']:.2f} pts/m³ ({difference_data['density_change_percent']:.2f}%)",
                'description': 'Change in point density',
                'impact': 'Scan quality or detail level'
            },
            {
                'name': 'Change Severity',
                'value': f"{difference_data['change_severity']:.2f}%",
                'description': 'Overall severity of changes',
                'impact': self._get_severity_impact(difference_data['change_severity'])
            }
        ]
        
        # Add rows to the table
        for metric in metrics:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            
            self.results_table.setItem(row, 0, QTableWidgetItem(metric['name']))
            self.results_table.setItem(row, 1, QTableWidgetItem(metric['value']))
            self.results_table.setItem(row, 2, QTableWidgetItem(metric['description']))
            self.results_table.setItem(row, 3, QTableWidgetItem(metric['impact']))
    
    def _get_severity_impact(self, severity):
        """Get impact description based on change severity"""
        if severity < 5:
            return "Minor structural changes"
        elif severity < 15:
            return "Moderate structural changes"
        else:
            return "Major structural changes"