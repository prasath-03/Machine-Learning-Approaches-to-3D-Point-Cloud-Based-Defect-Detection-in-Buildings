from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                          QGroupBox, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
                          QComboBox, QFormLayout, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QIcon

import numpy as np
import pandas as pd
import os

# Import core functionality modules
import utils_pyqt as utils


class DashboardWidget(QWidget):
    """Dashboard widget for BuildScan AI"""
    
    # Signal emitted when a file is loaded
    file_loaded = pyqtSignal(dict)
    
    def __init__(self, point_cloud_cache, parent=None):
        super().__init__(parent)
        self.point_cloud_cache = point_cloud_cache
        self.current_point_cloud = None
        
        # Set up the main layout
        main_layout = QVBoxLayout(self)
        
        # Create header section
        header_layout = QHBoxLayout()
        title_label = QLabel("BuildScan AI Dashboard")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Create import section
        import_group = QGroupBox("Import Point Cloud")
        import_layout = QVBoxLayout()
        
        # File selection
        file_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Point Cloud")
        self.load_btn.clicked.connect(self._load_point_cloud)
        
        # Add sample data button
        self.sample_btn = QPushButton("Generate Sample Data")
        self.sample_btn.clicked.connect(self._generate_sample_data)
        
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.sample_btn)
        
        import_layout.addLayout(file_layout)
        import_group.setLayout(import_layout)
        
        # Create point cloud info section
        info_group = QGroupBox("Point Cloud Information")
        info_layout = QVBoxLayout()
        
        self.info_table = QTableWidget(0, 2)
        self.info_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.info_table.verticalHeader().setVisible(False)
        
        info_layout.addWidget(self.info_table)
        info_group.setLayout(info_layout)
        
        # Create quick actions section
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout()
        
        # Buttons for actions
        self.view_btn = QPushButton("View in 3D Viewer")
        self.view_btn.clicked.connect(self._view_in_3d)
        self.view_btn.setEnabled(False)
        
        self.analyze_btn = QPushButton("Analyze Defects")
        self.analyze_btn.clicked.connect(self._analyze_defects)
        self.analyze_btn.setEnabled(False)
        
        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self._export_data)
        self.export_btn.setEnabled(False)
        
        actions_layout.addWidget(self.view_btn)
        actions_layout.addWidget(self.analyze_btn)
        actions_layout.addWidget(self.export_btn)
        actions_group.setLayout(actions_layout)
        
        # Add all sections to the main layout
        main_layout.addLayout(header_layout)
        main_layout.addWidget(import_group)
        main_layout.addWidget(info_group, stretch=2)
        main_layout.addWidget(actions_group)
    
    def _load_point_cloud(self):
        """Load a point cloud from a file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Point Cloud File",
            "",
            "Point Cloud Files (*.ply *.pcd *.xyz *.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load the point cloud
                point_cloud = utils.load_point_cloud(file_path)
                
                # Update cache with file name as key
                file_name = os.path.basename(file_path)
                self.point_cloud_cache.put(file_name, point_cloud)
                
                # Update current point cloud
                self.current_point_cloud = point_cloud
                
                # Update UI
                self._update_point_cloud_info(point_cloud, file_name)
                self._enable_actions(True)
                
                # Notify other widgets
                self.file_loaded.emit(point_cloud)
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load point cloud: {str(e)}")
    
    def _generate_sample_data(self):
        """Generate a sample point cloud for testing"""
        try:
            # Generate sample point cloud
            point_cloud = utils._generate_sample_point_cloud(num_points=5000)
            
            # Update cache
            self.point_cloud_cache.put("sample_data.ply", point_cloud)
            
            # Update current point cloud
            self.current_point_cloud = point_cloud
            
            # Update UI
            self._update_point_cloud_info(point_cloud, "sample_data.ply")
            self._enable_actions(True)
            
            # Notify other widgets
            self.file_loaded.emit(point_cloud)
            
            QMessageBox.information(self, "Success", "Sample point cloud generated successfully!")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate sample point cloud: {str(e)}")
    
    def _update_point_cloud_info(self, point_cloud, file_name):
        """Update the point cloud information table"""
        # Get point cloud statistics
        stats = utils.get_point_cloud_stats(point_cloud)
        
        # Clear the table
        self.info_table.setRowCount(0)
        
        # Add rows to the table
        self._add_info_row("File Name", file_name)
        self._add_info_row("Point Count", f"{stats['point_count']:,}")
        
        if stats['point_count'] > 0:
            dimensions = stats['dimensions']
            self._add_info_row("Dimensions (m)", f"X: {dimensions[0]:.2f}, Y: {dimensions[1]:.2f}, Z: {dimensions[2]:.2f}")
            
            center = stats['center']
            self._add_info_row("Center", f"X: {center[0]:.2f}, Y: {center[1]:.2f}, Z: {center[2]:.2f}")
            
            self._add_info_row("Approximate Area (m²)", f"{stats['approximate_area']:.2f}")
            
            has_colors = 'colors' in point_cloud and point_cloud['colors'] is not None
            self._add_info_row("Color Information", "Yes" if has_colors else "No")
            
            has_normals = 'normals' in point_cloud and point_cloud['normals'] is not None
            self._add_info_row("Normal Information", "Yes" if has_normals else "No")
    
    def _add_info_row(self, property_name, value):
        """Add a row to the information table"""
        row = self.info_table.rowCount()
        self.info_table.insertRow(row)
        
        property_item = QTableWidgetItem(property_name)
        property_item.setFlags(property_item.flags() & ~Qt.ItemIsEditable)
        
        value_item = QTableWidgetItem(str(value))
        value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
        
        self.info_table.setItem(row, 0, property_item)
        self.info_table.setItem(row, 1, value_item)
    
    def _enable_actions(self, enabled):
        """Enable or disable action buttons"""
        self.view_btn.setEnabled(enabled)
        self.analyze_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)
    
    def _view_in_3d(self):
        """Switch to 3D viewer tab"""
        if self.parent() and hasattr(self.parent(), 'central_widget'):
            # Find the index of the 3D viewer tab
            for i in range(self.parent().central_widget.count()):
                if self.parent().central_widget.tabText(i) == "3D Viewer":
                    self.parent().central_widget.setCurrentIndex(i)
                    break
    
    def _analyze_defects(self):
        """Switch to defect analysis tab"""
        if self.parent() and hasattr(self.parent(), 'central_widget'):
            # Find the index of the defect analysis tab
            for i in range(self.parent().central_widget.count()):
                if self.parent().central_widget.tabText(i) == "Defect Analysis":
                    self.parent().central_widget.setCurrentIndex(i)
                    break
    
    def _export_data(self):
        """Export point cloud data"""
        if not self.current_point_cloud:
            return
        
        # Create dialog to select export format
        formats = {
            "CSV (.csv)": ("csv", utils.export_to_csv),
            "PLY (.ply)": ("ply", utils.export_to_ply)
        }
        
        # Create a simple dialog with a dropdown
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Export Point Cloud")
        dialog.setText("Select the export format:")
        dialog.setIcon(QMessageBox.Question)
        
        # Add export format dropdown
        combo = QComboBox(dialog)
        for format_name in formats:
            combo.addItem(format_name)
        
        # Add combobox to the dialog
        layout = dialog.layout()
        layout.addWidget(combo, 1, 1)
        
        # Add buttons
        dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        
        # Show dialog
        result = dialog.exec_()
        
        if result == QMessageBox.Ok:
            selected_format = combo.currentText()
            extension, export_func = formats[selected_format]
            
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Point Cloud",
                f"export.{extension}",
                f"{selected_format} (*{extension})"
            )
            
            if file_path:
                try:
                    # Export the point cloud
                    if export_func(self.current_point_cloud, file_path=file_path):
                        QMessageBox.information(self, "Success", f"Point cloud exported successfully to {file_path}")
                    else:
                        QMessageBox.warning(self, "Warning", "Failed to export point cloud.")
                
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to export point cloud: {str(e)}")