#!/usr/bin/env python3
"""
BuildScan AI - Desktop Application Information

This script provides information about the BuildScan AI desktop application
when it cannot be launched in the current environment.
"""

import sys
import os

def main():
    """Print information about the desktop application"""
    print("=" * 80)
    print("BuildScan AI - Building Structure Analysis")
    print("=" * 80)
    print("\nThis is the desktop version of BuildScan AI, a 3D point cloud analysis")
    print("application for building structure defect detection.")
    print("\nThe PyQt5-based desktop application requires a desktop environment")
    print("with X11/Wayland support. It cannot run in this environment.")
    print("\nFeatures of the desktop application:")
    print("- 3D point cloud visualization and manipulation")
    print("- Machine learning-based defect detection")
    print("- Multiple detection algorithms (feature-based, clustering, segmentation)")
    print("- Comparative analysis between point cloud scans")
    print("- Historical tracking of building scans")
    print("- Report generation")
    print("\nThe Streamlit web interface provides a subset of these features")
    print("optimized for web browser access.")
    print("\nTo use the desktop application, please download the code and run it")
    print("on a system with PyQt5 properly installed.")

if __name__ == "__main__":
    main()