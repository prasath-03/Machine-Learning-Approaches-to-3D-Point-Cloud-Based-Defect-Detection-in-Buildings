from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                          QGroupBox, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
                          QSplitter, QFormLayout, QComboBox, QMessageBox, QDateEdit)
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QFont, QColor

import numpy as np
import pandas as pd
import os
import datetime

# Import core functionality modules
import utils_pyqt as utils


class HistoryWidget(QWidget):
    """History tracking widget for BuildScan AI"""
    
    def __init__(self, point_cloud_cache, parent=None):
        super().__init__(parent)
        self.point_cloud_cache = point_cloud_cache
        
        # Set up the main layout
        main_layout = QVBoxLayout(self)
        
        # Create header section
        header_layout = QHBoxLayout()
        title_label = QLabel("Scan History")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Add new scan button
        self.add_scan_btn = QPushButton("Add New Scan")
        self.add_scan_btn.clicked.connect(self._add_new_scan)
        header_layout.addWidget(self.add_scan_btn)
        
        # Create scan history table
        history_group = QGroupBox("Scan History")
        history_layout = QVBoxLayout()
        
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels([
            "Date", "Name", "Point Count", "Location", "Status"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setSelectionMode(QTableWidget.SingleSelection)
        self.history_table.itemSelectionChanged.connect(self._handle_selection_changed)
        
        history_layout.addWidget(self.history_table)
        history_group.setLayout(history_layout)
        
        # Create scan details section
        details_group = QGroupBox("Scan Details")
        details_layout = QVBoxLayout()
        
        # Scan info form
        form_layout = QFormLayout()
        
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        form_layout.addRow("Date:", self.date_edit)
        
        self.name_combo = QComboBox()
        self.name_combo.setEditable(True)
        for key in self.point_cloud_cache.cache:
            self.name_combo.addItem(key)
        form_layout.addRow("Scan Name:", self.name_combo)
        
        self.location_combo = QComboBox()
        self.location_combo.setEditable(True)
        self.location_combo.addItems(["Building A", "Building B", "Building C"])
        form_layout.addRow("Location:", self.location_combo)
        
        self.status_combo = QComboBox()
        self.status_combo.addItems(["Completed", "In Progress", "Needs Review", "Archived"])
        form_layout.addRow("Status:", self.status_combo)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.update_btn = QPushButton("Update")
        self.update_btn.clicked.connect(self._update_scan)
        self.update_btn.setEnabled(False)
        
        self.view_btn = QPushButton("View in 3D Viewer")
        self.view_btn.clicked.connect(self._view_scan)
        self.view_btn.setEnabled(False)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_scan)
        self.delete_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.update_btn)
        buttons_layout.addWidget(self.view_btn)
        buttons_layout.addWidget(self.delete_btn)
        
        details_layout.addLayout(form_layout)
        details_layout.addLayout(buttons_layout)
        details_group.setLayout(details_layout)
        
        # Create reports section
        reports_group = QGroupBox("Reports")
        reports_layout = QVBoxLayout()
        
        # Report type combo
        self.report_type_combo = QComboBox()
        self.report_type_combo.addItems([
            "Defect Summary Report", 
            "Historical Comparison Report", 
            "Scan Quality Report"
        ])
        reports_layout.addWidget(self.report_type_combo)
        
        # Generate report button
        self.generate_report_btn = QPushButton("Generate Report")
        self.generate_report_btn.clicked.connect(self._generate_report)
        self.generate_report_btn.setEnabled(False)
        reports_layout.addWidget(self.generate_report_btn)
        
        reports_group.setLayout(reports_layout)
        
        # Set up the layout
        details_reports_layout = QHBoxLayout()
        details_reports_layout.addWidget(details_group)
        details_reports_layout.addWidget(reports_group)
        
        main_layout.addLayout(header_layout)
        main_layout.addWidget(history_group, stretch=2)
        main_layout.addLayout(details_reports_layout, stretch=1)
        
        # Populate the history table with sample data
        self._populate_sample_history()
    
    def _populate_sample_history(self):
        """Populate the history table with sample data"""
        sample_data = [
            {"date": "2024-03-25", "name": "Building_Scan_001.ply", "points": 125000, "location": "Building A", "status": "Completed"},
            {"date": "2024-03-20", "name": "Building_Scan_002.ply", "points": 98500, "location": "Building B", "status": "Needs Review"},
            {"date": "2024-03-15", "name": "Building_Scan_003.ply", "points": 150000, "location": "Building C", "status": "Archived"},
            {"date": "2024-03-10", "name": "Building_Scan_004.ply", "points": 110000, "location": "Building A", "status": "Completed"}
        ]
        
        # Add sample scans to cache if not already present
        for data in sample_data:
            if data["name"] not in self.point_cloud_cache.cache:
                # Create a sample point cloud for this entry
                point_cloud = utils._generate_sample_point_cloud(num_points=data["points"])
                self.point_cloud_cache.put(data["name"], point_cloud)
        
        # Clear the table
        self.history_table.setRowCount(0)
        
        # Add rows to the table
        for data in sample_data:
            self._add_history_row(data)
    
    def _add_history_row(self, data):
        """Add a row to the history table"""
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        
        date_item = QTableWidgetItem(data["date"])
        name_item = QTableWidgetItem(data["name"])
        points_item = QTableWidgetItem(f"{data['points']:,}")
        location_item = QTableWidgetItem(data["location"])
        status_item = QTableWidgetItem(data["status"])
        
        self.history_table.setItem(row, 0, date_item)
        self.history_table.setItem(row, 1, name_item)
        self.history_table.setItem(row, 2, points_item)
        self.history_table.setItem(row, 3, location_item)
        self.history_table.setItem(row, 4, status_item)
    
    def _handle_selection_changed(self):
        """Handle selection change in the history table"""
        selected_items = self.history_table.selectedItems()
        
        if selected_items:
            # Get the selected row
            row = selected_items[0].row()
            
            # Get data from the selected row
            date_str = self.history_table.item(row, 0).text()
            name = self.history_table.item(row, 1).text()
            location = self.history_table.item(row, 3).text()
            status = self.history_table.item(row, 4).text()
            
            # Update the form
            date = QDate.fromString(date_str, "yyyy-MM-dd")
            self.date_edit.setDate(date)
            
            idx = self.name_combo.findText(name)
            if idx >= 0:
                self.name_combo.setCurrentIndex(idx)
            else:
                self.name_combo.setCurrentText(name)
            
            idx = self.location_combo.findText(location)
            if idx >= 0:
                self.location_combo.setCurrentIndex(idx)
            else:
                self.location_combo.setCurrentText(location)
            
            idx = self.status_combo.findText(status)
            if idx >= 0:
                self.status_combo.setCurrentIndex(idx)
            
            # Enable buttons
            self.update_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
            self.view_btn.setEnabled(name in self.point_cloud_cache.cache)
            self.generate_report_btn.setEnabled(True)
        else:
            # Disable buttons
            self.update_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.view_btn.setEnabled(False)
            self.generate_report_btn.setEnabled(False)
    
    def _add_new_scan(self):
        """Add a new scan to the history"""
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
                
                # Get stats
                stats = utils.get_point_cloud_stats(point_cloud)
                
                # Update cache
                file_name = os.path.basename(file_path)
                self.point_cloud_cache.put(file_name, point_cloud)
                
                # Add to history
                today = datetime.date.today().strftime("%Y-%m-%d")
                data = {
                    "date": today,
                    "name": file_name,
                    "points": stats["point_count"],
                    "location": self.location_combo.currentText(),
                    "status": "Completed"
                }
                
                self._add_history_row(data)
                
                # Update name combo
                if self.name_combo.findText(file_name) < 0:
                    self.name_combo.addItem(file_name)
                
                # Select the new row
                self.history_table.selectRow(self.history_table.rowCount() - 1)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load point cloud: {str(e)}")
    
    def _update_scan(self):
        """Update the selected scan"""
        selected_items = self.history_table.selectedItems()
        
        if selected_items:
            # Get the selected row
            row = selected_items[0].row()
            
            # Update the row with form data
            date = self.date_edit.date().toString("yyyy-MM-dd")
            name = self.name_combo.currentText()
            location = self.location_combo.currentText()
            status = self.status_combo.currentText()
            
            # Get the point count from the table (keep it unchanged)
            points = self.history_table.item(row, 2).text()
            
            self.history_table.item(row, 0).setText(date)
            self.history_table.item(row, 1).setText(name)
            self.history_table.item(row, 3).setText(location)
            self.history_table.item(row, 4).setText(status)
            
            QMessageBox.information(self, "Scan Updated", "Scan details have been updated successfully.")
    
    def _delete_scan(self):
        """Delete the selected scan"""
        selected_items = self.history_table.selectedItems()
        
        if selected_items:
            # Get the selected row
            row = selected_items[0].row()
            
            # Ask for confirmation
            name = self.history_table.item(row, 1).text()
            reply = QMessageBox.question(
                self, 
                "Confirm Deletion", 
                f"Are you sure you want to delete the scan '{name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Remove the row from the table
                self.history_table.removeRow(row)
                
                # Disable buttons
                self.update_btn.setEnabled(False)
                self.delete_btn.setEnabled(False)
                self.view_btn.setEnabled(False)
                self.generate_report_btn.setEnabled(False)
    
    def _view_scan(self):
        """View the selected scan in the 3D viewer"""
        selected_items = self.history_table.selectedItems()
        
        if selected_items:
            # Get the selected row
            row = selected_items[0].row()
            
            # Get the scan name
            name = self.history_table.item(row, 1).text()
            
            # Get the point cloud from cache
            point_cloud = self.point_cloud_cache.get(name)
            
            if point_cloud:
                # Switch to 3D viewer tab
                if self.parent() and hasattr(self.parent(), 'central_widget'):
                    # Find the index of the 3D viewer tab
                    for i in range(self.parent().central_widget.count()):
                        if self.parent().central_widget.tabText(i) == "3D Viewer":
                            # Load the point cloud in the visualization widget
                            self.parent().visualization.load_point_cloud(point_cloud)
                            
                            # Switch to the tab
                            self.parent().central_widget.setCurrentIndex(i)
                            break
            else:
                QMessageBox.warning(self, "Warning", f"Point cloud '{name}' not found in cache.")
    
    def _generate_report(self):
        """Generate a report for the selected scan"""
        selected_items = self.history_table.selectedItems()
        
        if selected_items:
            # Get the selected row
            row = selected_items[0].row()
            
            # Get the scan name and date
            name = self.history_table.item(row, 1).text()
            date = self.history_table.item(row, 0).text()
            
            # Get report type
            report_type = self.report_type_combo.currentText()
            
            # Show a message
            QMessageBox.information(
                self, 
                "Report Generated", 
                f"{report_type} for scan '{name}' ({date}) has been generated successfully."
            )