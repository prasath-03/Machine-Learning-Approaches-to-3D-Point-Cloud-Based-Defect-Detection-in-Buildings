from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                          QGroupBox, QCheckBox, QSlider, QSpinBox, QDoubleSpinBox,
                          QTabWidget, QComboBox, QFormLayout, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor


class SettingsWidget(QWidget):
    """Settings widget for BuildScan AI"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up the main layout
        main_layout = QVBoxLayout(self)
        
        # Create header section
        header_layout = QHBoxLayout()
        title_label = QLabel("Application Settings")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Add settings tabs
        settings_tabs = QTabWidget()
        
        # Application settings tab
        app_tab = QWidget()
        app_layout = QVBoxLayout(app_tab)
        
        # UI settings group
        ui_group = QGroupBox("User Interface")
        ui_layout = QFormLayout()
        
        # Theme selector
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "System"])
        ui_layout.addRow("Theme:", self.theme_combo)
        
        # Font size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setMinimum(8)
        self.font_size_spin.setMaximum(18)
        self.font_size_spin.setValue(10)
        ui_layout.addRow("Font Size:", self.font_size_spin)
        
        ui_group.setLayout(ui_layout)
        
        # Performance settings group
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout()
        
        # Point cloud size limit
        self.point_limit_spin = QSpinBox()
        self.point_limit_spin.setMinimum(10000)
        self.point_limit_spin.setMaximum(10000000)
        self.point_limit_spin.setSingleStep(10000)
        self.point_limit_spin.setValue(1000000)
        perf_layout.addRow("Point Cloud Size Limit:", self.point_limit_spin)
        
        # Cache size
        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setMinimum(1)
        self.cache_size_spin.setMaximum(20)
        self.cache_size_spin.setValue(5)
        perf_layout.addRow("Cache Size (# of point clouds):", self.cache_size_spin)
        
        # Downsampling enabled checkbox
        self.downsampling_check = QCheckBox("Enable automatic downsampling")
        self.downsampling_check.setChecked(True)
        perf_layout.addRow(self.downsampling_check)
        
        perf_group.setLayout(perf_layout)
        
        # Add groups to the tab
        app_layout.addWidget(ui_group)
        app_layout.addWidget(perf_group)
        app_layout.addStretch()
        
        # Add save button
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        app_layout.addWidget(save_btn)
        
        # Analysis settings tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        
        # Defect detection settings group
        detection_group = QGroupBox("Defect Detection")
        detection_layout = QFormLayout()
        
        # Default model
        self.default_model_combo = QComboBox()
        self.default_model_combo.addItems(["Random Forest", "SVM", "Isolation Forest"])
        detection_layout.addRow("Default Model:", self.default_model_combo)
        
        # Default threshold
        self.detection_threshold_spin = QDoubleSpinBox()
        self.detection_threshold_spin.setMinimum(0.01)
        self.detection_threshold_spin.setMaximum(0.99)
        self.detection_threshold_spin.setSingleStep(0.01)
        self.detection_threshold_spin.setValue(0.5)
        detection_layout.addRow("Default Detection Threshold:", self.detection_threshold_spin)
        
        detection_group.setLayout(detection_layout)
        
        # Visualization settings group
        viz_group = QGroupBox("Visualization")
        viz_layout = QFormLayout()
        
        # Default colormap
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "jet", "rainbow"])
        viz_layout.addRow("Default Colormap:", self.colormap_combo)
        
        # Default point size
        self.point_size_spin = QDoubleSpinBox()
        self.point_size_spin.setMinimum(0.1)
        self.point_size_spin.setMaximum(10.0)
        self.point_size_spin.setSingleStep(0.1)
        self.point_size_spin.setValue(1.0)
        viz_layout.addRow("Default Point Size:", self.point_size_spin)
        
        viz_group.setLayout(viz_layout)
        
        # Preprocessing settings group
        preproc_group = QGroupBox("Preprocessing")
        preproc_layout = QFormLayout()
        
        # Voxel size for downsampling
        self.voxel_size_spin = QDoubleSpinBox()
        self.voxel_size_spin.setMinimum(0.001)
        self.voxel_size_spin.setMaximum(1.0)
        self.voxel_size_spin.setSingleStep(0.001)
        self.voxel_size_spin.setValue(0.05)
        preproc_layout.addRow("Voxel Size for Downsampling:", self.voxel_size_spin)
        
        # Outlier removal enabled
        self.outlier_removal_check = QCheckBox("Enable outlier removal")
        self.outlier_removal_check.setChecked(True)
        preproc_layout.addRow(self.outlier_removal_check)
        
        # Normal computation enabled
        self.compute_normals_check = QCheckBox("Compute normals if missing")
        self.compute_normals_check.setChecked(True)
        preproc_layout.addRow(self.compute_normals_check)
        
        preproc_group.setLayout(preproc_layout)
        
        # Add groups to the tab
        analysis_layout.addWidget(detection_group)
        analysis_layout.addWidget(viz_group)
        analysis_layout.addWidget(preproc_group)
        analysis_layout.addStretch()
        
        # Add save button
        save_analysis_btn = QPushButton("Save Analysis Settings")
        save_analysis_btn.clicked.connect(self._save_analysis_settings)
        analysis_layout.addWidget(save_analysis_btn)
        
        # Add tabs to the tab widget
        settings_tabs.addTab(app_tab, "Application")
        settings_tabs.addTab(analysis_tab, "Analysis")
        
        # Add all sections to the main layout
        main_layout.addLayout(header_layout)
        main_layout.addWidget(settings_tabs)
    
    def _save_settings(self):
        """Save application settings"""
        # In a real implementation, this would save settings to a file
        # For now, just show a message
        QMessageBox.information(self, "Settings Saved", "Application settings have been saved.")
    
    def _save_analysis_settings(self):
        """Save analysis settings"""
        # In a real implementation, this would save settings to a file
        # For now, just show a message
        QMessageBox.information(self, "Settings Saved", "Analysis settings have been saved.")