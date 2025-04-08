from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                          QGroupBox, QRadioButton, QSlider, QSpinBox, QDoubleSpinBox,
                          QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                          QSplitter, QFormLayout, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

import numpy as np
import pandas as pd
import os

# Import core functionality modules
import utils_pyqt as utils
import defect_detection_pyqt as defect_detection


class DefectDetectionWidget(QWidget):
    """Defect detection widget for BuildScan AI"""
    
    # Signal emitted when defects are detected
    defects_detected = pyqtSignal(np.ndarray)
    
    def __init__(self, point_cloud_cache, parent=None):
        super().__init__(parent)
        self.point_cloud_cache = point_cloud_cache
        self.current_point_cloud = None
        self.current_defect_indices = None
        
        # Set up the main layout
        main_layout = QHBoxLayout(self)
        
        # Create the left panel for controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Title
        title_label = QLabel("Defect Detection")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        control_layout.addWidget(title_label)
        
        # Create tabs for different detection methods
        self.method_tabs = QTabWidget()
        
        # Feature-based detection tab
        feature_tab = QWidget()
        feature_layout = QVBoxLayout(feature_tab)
        
        # Model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        
        self.rf_radio = QRadioButton("Random Forest")
        self.rf_radio.setChecked(True)
        self.svm_radio = QRadioButton("Support Vector Machine")
        self.isolation_radio = QRadioButton("Isolation Forest")
        
        model_layout.addWidget(self.rf_radio)
        model_layout.addWidget(self.svm_radio)
        model_layout.addWidget(self.isolation_radio)
        model_group.setLayout(model_layout)
        
        # Feature selection group
        feature_group = QGroupBox("Feature Selection")
        feature_layout = QVBoxLayout()
        
        self.normals_check = QRadioButton("Normals")
        self.normals_check.setChecked(True)
        self.curvature_check = QRadioButton("Curvature")
        self.roughness_check = QRadioButton("Roughness")
        self.verticality_check = QRadioButton("Verticality")
        
        feature_layout.addWidget(self.normals_check)
        feature_layout.addWidget(self.curvature_check)
        feature_layout.addWidget(self.roughness_check)
        feature_layout.addWidget(self.verticality_check)
        feature_group.setLayout(feature_layout)
        
        # Run button
        self.run_feature_btn = QPushButton("Run Feature-Based Detection")
        self.run_feature_btn.clicked.connect(self._run_feature_based_detection)
        self.run_feature_btn.setEnabled(False)
        
        feature_layout.addWidget(model_group)
        feature_layout.addWidget(feature_group)
        feature_layout.addWidget(self.run_feature_btn)
        feature_layout.addStretch()
        
        # Clustering-based detection tab
        clustering_tab = QWidget()
        clustering_layout = QVBoxLayout(clustering_tab)
        
        # Method selection group
        method_group = QGroupBox("Clustering Method")
        method_layout = QVBoxLayout()
        
        self.dbscan_radio = QRadioButton("DBSCAN")
        self.dbscan_radio.setChecked(True)
        self.dbscan_radio.toggled.connect(lambda: self._update_clustering_ui("dbscan"))
        
        self.kmeans_radio = QRadioButton("K-Means")
        self.kmeans_radio.toggled.connect(lambda: self._update_clustering_ui("kmeans"))
        
        method_layout.addWidget(self.dbscan_radio)
        method_layout.addWidget(self.kmeans_radio)
        method_group.setLayout(method_layout)
        
        # Parameters group
        params_group = QGroupBox("Clustering Parameters")
        self.params_layout = QFormLayout()
        
        # DBSCAN parameters
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setMinimum(0.01)
        self.eps_spin.setMaximum(1.0)
        self.eps_spin.setSingleStep(0.01)
        self.eps_spin.setValue(0.1)
        self.params_layout.addRow("Epsilon:", self.eps_spin)
        
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setMinimum(1)
        self.min_samples_spin.setMaximum(100)
        self.min_samples_spin.setValue(10)
        self.params_layout.addRow("Min Samples:", self.min_samples_spin)
        
        # K-Means parameters (initially hidden)
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setMinimum(2)
        self.n_clusters_spin.setMaximum(20)
        self.n_clusters_spin.setValue(5)
        self.params_layout.addRow("Number of Clusters:", self.n_clusters_spin)
        self.n_clusters_spin.setVisible(False)
        self.params_layout.labelForField(self.n_clusters_spin).setVisible(False)
        
        params_group.setLayout(self.params_layout)
        
        # Run button
        self.run_clustering_btn = QPushButton("Run Clustering-Based Detection")
        self.run_clustering_btn.clicked.connect(self._run_clustering_detection)
        self.run_clustering_btn.setEnabled(False)
        
        clustering_layout.addWidget(method_group)
        clustering_layout.addWidget(params_group)
        clustering_layout.addWidget(self.run_clustering_btn)
        clustering_layout.addStretch()
        
        # Segmentation-based detection tab
        segmentation_tab = QWidget()
        segmentation_layout = QVBoxLayout(segmentation_tab)
        
        # Method selection group
        seg_method_group = QGroupBox("Segmentation Method")
        seg_method_layout = QVBoxLayout()
        
        self.region_growing_radio = QRadioButton("Region Growing")
        self.region_growing_radio.setChecked(True)
        
        self.ransac_radio = QRadioButton("RANSAC")
        
        seg_method_layout.addWidget(self.region_growing_radio)
        seg_method_layout.addWidget(self.ransac_radio)
        seg_method_group.setLayout(seg_method_layout)
        
        # Parameters group
        seg_params_group = QGroupBox("Segmentation Parameters")
        seg_params_layout = QFormLayout()
        
        # Region growing parameters
        self.smoothness_spin = QDoubleSpinBox()
        self.smoothness_spin.setMinimum(0.1)
        self.smoothness_spin.setMaximum(10.0)
        self.smoothness_spin.setSingleStep(0.1)
        self.smoothness_spin.setValue(3.0)
        seg_params_layout.addRow("Smoothness Threshold:", self.smoothness_spin)
        
        self.min_cluster_spin = QSpinBox()
        self.min_cluster_spin.setMinimum(10)
        self.min_cluster_spin.setMaximum(1000)
        self.min_cluster_spin.setValue(100)
        seg_params_layout.addRow("Min Cluster Size:", self.min_cluster_spin)
        
        # RANSAC parameters
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setMinimum(0.01)
        self.distance_spin.setMaximum(1.0)
        self.distance_spin.setSingleStep(0.01)
        self.distance_spin.setValue(0.05)
        seg_params_layout.addRow("Distance Threshold:", self.distance_spin)
        
        seg_params_group.setLayout(seg_params_layout)
        
        # Run button
        self.run_segmentation_btn = QPushButton("Run Segmentation-Based Detection")
        self.run_segmentation_btn.clicked.connect(self._run_segmentation_detection)
        self.run_segmentation_btn.setEnabled(False)
        
        segmentation_layout.addWidget(seg_method_group)
        segmentation_layout.addWidget(seg_params_group)
        segmentation_layout.addWidget(self.run_segmentation_btn)
        segmentation_layout.addStretch()
        
        # Add tabs to the tab widget
        self.method_tabs.addTab(feature_tab, "Feature-Based")
        self.method_tabs.addTab(clustering_tab, "Clustering")
        self.method_tabs.addTab(segmentation_tab, "Segmentation")
        
        control_layout.addWidget(self.method_tabs)
        
        # Add a spacer at the bottom
        control_layout.addStretch()
        
        # Create the right panel for results
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        
        # Results table
        self.results_group = QGroupBox("Detection Results")
        results_table_layout = QVBoxLayout()
        
        self.results_table = QTableWidget(0, 2)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        results_table_layout.addWidget(self.results_table)
        self.results_group.setLayout(results_table_layout)
        
        # Visualization controls
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()
        
        self.view_btn = QPushButton("View Defects in 3D Viewer")
        self.view_btn.clicked.connect(self._view_defects)
        self.view_btn.setEnabled(False)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        
        viz_layout.addWidget(self.view_btn)
        viz_layout.addWidget(self.export_btn)
        viz_group.setLayout(viz_layout)
        
        results_layout.addWidget(self.results_group, stretch=3)
        results_layout.addWidget(viz_group)
        
        # Create a splitter to allow resizing of panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(results_panel)
        splitter.setSizes([400, 600])  # Initial sizes
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
    
    def load_point_cloud(self, point_cloud):
        """Load a point cloud for analysis
        
        Args:
            point_cloud (dict): Point cloud data
        """
        self.current_point_cloud = point_cloud
        self.current_defect_indices = None
        
        # Enable/disable buttons based on point cloud availability
        enabled = point_cloud is not None
        self.run_feature_btn.setEnabled(enabled)
        self.run_clustering_btn.setEnabled(enabled)
        self.run_segmentation_btn.setEnabled(enabled)
        
        # Clear results
        self.results_table.setRowCount(0)
        self.view_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
    
    def _run_defect_detection(self):
        """Run defect detection based on the current tab settings"""
        if not self.current_point_cloud:
            return
        
        # Determine which tab is active
        current_tab = self.method_tabs.currentWidget()
        tab_index = self.method_tabs.currentIndex()
        
        if tab_index == 0:
            self._run_feature_based_detection()
        elif tab_index == 1:
            self._run_clustering_detection()
        elif tab_index == 2:
            self._run_segmentation_detection()
    
    def _run_feature_based_detection(self):
        """Run feature-based defect detection"""
        if not self.current_point_cloud:
            return
        
        try:
            # Determine model type
            if self.rf_radio.isChecked():
                model_type = "Random Forest"
            elif self.svm_radio.isChecked():
                model_type = "SVM"
            else:
                model_type = "Isolation Forest"
            
            # Determine feature set
            feature_set = []
            if self.normals_check.isChecked():
                feature_set.append("Normals")
            if self.curvature_check.isChecked():
                feature_set.append("Curvature")
            if self.roughness_check.isChecked():
                feature_set.append("Roughness")
            if self.verticality_check.isChecked():
                feature_set.append("Verticality")
            
            # Run detection
            labels, defect_indices, importance_dict, report = defect_detection.detect_defects_features(
                self.current_point_cloud,
                model_type=model_type,
                feature_set=feature_set
            )
            
            # Update results table
            self._update_results_table(defect_indices, report)
            
            # Store defect indices
            self.current_defect_indices = defect_indices
            
            # Enable visualization buttons
            self.view_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
            # Emit defects detected signal
            self.defects_detected.emit(defect_indices)
        
        except Exception as e:
            self._show_error(f"Error during feature-based detection: {str(e)}")
    
    def _run_clustering_detection(self):
        """Run clustering-based defect detection"""
        if not self.current_point_cloud:
            return
        
        try:
            # Determine clustering method
            method = "dbscan" if self.dbscan_radio.isChecked() else "kmeans"
            
            # Get parameters
            eps = self.eps_spin.value()
            min_samples = self.min_samples_spin.value()
            n_clusters = self.n_clusters_spin.value()
            
            # Run detection
            cluster_labels, defect_indices = defect_detection.detect_defects_clustering(
                self.current_point_cloud,
                method=method,
                eps=eps,
                min_samples=min_samples,
                n_clusters=n_clusters
            )
            
            # Update results table
            self._update_results_table_clustering(cluster_labels, defect_indices)
            
            # Store defect indices
            self.current_defect_indices = defect_indices
            
            # Enable visualization buttons
            self.view_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
            # Emit defects detected signal
            self.defects_detected.emit(defect_indices)
        
        except Exception as e:
            self._show_error(f"Error during clustering-based detection: {str(e)}")
    
    def _run_segmentation_detection(self):
        """Run segmentation-based defect detection"""
        if not self.current_point_cloud:
            return
        
        try:
            # Determine segmentation method
            method = "region_growing" if self.region_growing_radio.isChecked() else "ransac"
            
            # Get parameters
            smoothness_threshold = self.smoothness_spin.value()
            min_cluster_size = self.min_cluster_spin.value()
            distance_threshold = self.distance_spin.value()
            
            # Run detection
            segment_labels, defect_indices = defect_detection.detect_defects_segmentation(
                self.current_point_cloud,
                method=method,
                smoothness_threshold=smoothness_threshold,
                min_cluster_size=min_cluster_size,
                distance_threshold=distance_threshold
            )
            
            # Update results table
            self._update_results_table_segmentation(segment_labels, defect_indices)
            
            # Store defect indices
            self.current_defect_indices = defect_indices
            
            # Enable visualization buttons
            self.view_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
            # Emit defects detected signal
            self.defects_detected.emit(defect_indices)
        
        except Exception as e:
            self._show_error(f"Error during segmentation-based detection: {str(e)}")
    
    def _update_clustering_ui(self, method):
        """Update UI elements based on selected clustering method"""
        if method == "dbscan":
            # Show DBSCAN parameters, hide K-Means parameters
            self.eps_spin.setVisible(True)
            self.min_samples_spin.setVisible(True)
            self.params_layout.labelForField(self.eps_spin).setVisible(True)
            self.params_layout.labelForField(self.min_samples_spin).setVisible(True)
            
            self.n_clusters_spin.setVisible(False)
            self.params_layout.labelForField(self.n_clusters_spin).setVisible(False)
        else:
            # Show K-Means parameters, hide DBSCAN parameters
            self.eps_spin.setVisible(False)
            self.min_samples_spin.setVisible(False)
            self.params_layout.labelForField(self.eps_spin).setVisible(False)
            self.params_layout.labelForField(self.min_samples_spin).setVisible(False)
            
            self.n_clusters_spin.setVisible(True)
            self.params_layout.labelForField(self.n_clusters_spin).setVisible(True)
    
    def _update_results_table(self, defect_indices, report):
        """Update the results table with defect detection results
        
        Args:
            defect_indices (numpy.ndarray): Indices of detected defects
            report (str): Classification report
        """
        # Clear the table
        self.results_table.setRowCount(0)
        
        if defect_indices is None or len(defect_indices) == 0:
            self._add_result_row("Status", "No defects detected")
            return
        
        # Add basic results
        self._add_result_row("Total Points", str(len(self.current_point_cloud['points'])))
        self._add_result_row("Defects Detected", str(len(defect_indices)))
        
        defect_percent = (len(defect_indices) / len(self.current_point_cloud['points'])) * 100
        self._add_result_row("Defect Percentage", f"{defect_percent:.2f}%")
        
        # Add classification report metrics
        if isinstance(report, dict):
            if '1' in report:  # Defect class
                self._add_result_row("Defect Precision", f"{report['1']['precision']:.4f}")
                self._add_result_row("Defect Recall", f"{report['1']['recall']:.4f}")
                self._add_result_row("Defect F1-Score", f"{report['1']['f1-score']:.4f}")
            
            if '0' in report:  # Normal class
                self._add_result_row("Normal Precision", f"{report['0']['precision']:.4f}")
                self._add_result_row("Normal Recall", f"{report['0']['recall']:.4f}")
                self._add_result_row("Normal F1-Score", f"{report['0']['f1-score']:.4f}")
            
            if 'accuracy' in report:
                self._add_result_row("Accuracy", f"{report['accuracy']:.4f}")
        
        # Add severity assessment based on defect percentage
        if defect_percent < 5:
            severity = "Low"
        elif defect_percent < 15:
            severity = "Medium"
        else:
            severity = "High"
        
        self._add_result_row("Defect Severity", severity)
    
    def _update_results_table_clustering(self, cluster_labels, defect_indices):
        """Update results table with clustering results"""
        # Clear the table
        self.results_table.setRowCount(0)
        
        if defect_indices is None or len(defect_indices) == 0:
            self._add_result_row("Status", "No defects detected")
            return
        
        # Add basic results
        self._add_result_row("Total Points", str(len(self.current_point_cloud['points'])))
        self._add_result_row("Defects Detected", str(len(defect_indices)))
        
        # Count unique clusters
        if cluster_labels is not None:
            num_clusters = len(np.unique(cluster_labels))
            self._add_result_row("Number of Clusters", str(num_clusters))
        
        defect_percent = (len(defect_indices) / len(self.current_point_cloud['points'])) * 100
        self._add_result_row("Defect Percentage", f"{defect_percent:.2f}%")
        
        # Add severity assessment based on defect percentage
        if defect_percent < 5:
            severity = "Low"
        elif defect_percent < 15:
            severity = "Medium"
        else:
            severity = "High"
        
        self._add_result_row("Defect Severity", severity)
    
    def _update_results_table_segmentation(self, segment_labels, defect_indices):
        """Update results table with segmentation results"""
        # Clear the table
        self.results_table.setRowCount(0)
        
        if defect_indices is None or len(defect_indices) == 0:
            self._add_result_row("Status", "No defects detected")
            return
        
        # Add basic results
        self._add_result_row("Total Points", str(len(self.current_point_cloud['points'])))
        self._add_result_row("Defects Detected", str(len(defect_indices)))
        
        defect_percent = (len(defect_indices) / len(self.current_point_cloud['points'])) * 100
        self._add_result_row("Defect Percentage", f"{defect_percent:.2f}%")
        
        # Add severity assessment based on defect percentage
        if defect_percent < 5:
            severity = "Low"
        elif defect_percent < 15:
            severity = "Medium"
        else:
            severity = "High"
        
        self._add_result_row("Defect Severity", severity)
    
    def _add_result_row(self, metric, value):
        """Add a row to the results table"""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        metric_item = QTableWidgetItem(metric)
        metric_item.setFlags(metric_item.flags() & ~Qt.ItemIsEditable)
        
        value_item = QTableWidgetItem(value)
        value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
        
        self.results_table.setItem(row, 0, metric_item)
        self.results_table.setItem(row, 1, value_item)
    
    def _view_defects(self):
        """View defects in the 3D viewer tab"""
        if self.current_defect_indices is not None:
            # Switch to 3D viewer tab
            if self.parent() and hasattr(self.parent(), 'central_widget'):
                # Find the index of the 3D viewer tab
                for i in range(self.parent().central_widget.count()):
                    if self.parent().central_widget.tabText(i) == "3D Viewer":
                        self.parent().central_widget.setCurrentIndex(i)
                        break
    
    def _export_results(self):
        """Export defect detection results"""
        if not self.current_point_cloud or self.current_defect_indices is None:
            return
        
        try:
            # Export to CSV
            success = utils.export_to_csv(
                self.current_point_cloud,
                defect_indices=self.current_defect_indices,
                file_path="defect_detection_results.csv"
            )
            
            if success:
                self._show_info("Results exported successfully to defect_detection_results.csv")
            else:
                self._show_error("Failed to export results")
        
        except Exception as e:
            self._show_error(f"Error exporting results: {str(e)}")
    
    def _show_error(self, message):
        """Show an error message"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Error", message)
    
    def _show_info(self, message):
        """Show an information message"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "Information", message)