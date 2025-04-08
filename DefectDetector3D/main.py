import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QStackedWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont

# Import UI components
from ui.dashboard import DashboardWidget
from ui.defect_detection import DefectDetectionWidget
from ui.visualization import VisualizationWidget
from ui.comparative_analysis import ComparativeAnalysisWidget
from ui.settings import SettingsWidget
from ui.history import HistoryWidget

# Import core functionality modules - use PyQt-compatible versions
import utils_pyqt as utils
import defect_detection_pyqt as defect_detection
import visualization_pyqt as visualization


class MainWindow(QMainWindow):
    """Main application window for BuildScan AI"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize the UI
        self.setWindowTitle("BuildScan AI - Structural Defect Detection")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)
        
        # Initialize data cache
        self.point_cloud_cache = utils.PointCloudCache()
        
        # Set up the central widget with tab navigation
        self.central_widget = QTabWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create tab widgets
        self.dashboard = DashboardWidget(self.point_cloud_cache)
        self.visualization = VisualizationWidget(self.point_cloud_cache)
        self.defect_detection = DefectDetectionWidget(self.point_cloud_cache)
        self.comparative_analysis = ComparativeAnalysisWidget(self.point_cloud_cache)
        self.history = HistoryWidget(self.point_cloud_cache)
        self.settings = SettingsWidget()
        
        # Add tabs
        self.central_widget.addTab(self.dashboard, "Dashboard")
        self.central_widget.addTab(self.visualization, "3D Viewer")
        self.central_widget.addTab(self.defect_detection, "Defect Analysis")
        self.central_widget.addTab(self.comparative_analysis, "Comparative Analysis")
        self.central_widget.addTab(self.history, "History")
        self.central_widget.addTab(self.settings, "Settings")
        
        # Connect signals between widgets
        self._connect_signals()
        
        # Apply CSS styling
        self._apply_styling()
    
    def _connect_signals(self):
        """Connect signals between widgets for data communication"""
        # Example: when a file is loaded in dashboard, update the visualization
        self.dashboard.file_loaded.connect(self.visualization.load_point_cloud)
        self.dashboard.file_loaded.connect(self.defect_detection.load_point_cloud)
        
        # When defects are detected, update visualization
        self.defect_detection.defects_detected.connect(self.visualization.highlight_defects)
    
    def _apply_styling(self):
        """Apply CSS styling to the application"""
        # Set application font
        app_font = QFont("Segoe UI", 10)
        QApplication.setFont(app_font)


def main():
    """Application entry point"""
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("=" * 80)
        print("BuildScan AI - Building Structure Analysis")
        print("=" * 80)
        print("\nERROR: Could not start GUI application in this environment.")
        print(f"Reason: {str(e)}")
        print("\nThis application requires a desktop environment with X11/Wayland support.")
        print("Please run this on a desktop system with PyQt5 properly installed.")
        print("\nFallback mode: Using Streamlit web interface.")
        print("Visit the Streamlit URL to use the web interface version.")
        print("\nNote: The Streamlit version provides a subset of the desktop application's")
        print("functionality, optimized for web browser access.")
        print("\nAlternatively, you can download this code and run it on your local machine")
        print("with PyQt5 installed to use the full desktop application.")


if __name__ == "__main__":
    main()