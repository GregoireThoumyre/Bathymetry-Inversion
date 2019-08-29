# -*- coding: utf-8 -*-
import sys
import context
import qdarkstyle
from PyQt5 import QtWidgets

from src.graphics import ApplicationWindow

"""
File: window.py
Description: GUI of the main project. Allow the user to display timestacks,
bathymetries and predicted bathymetries. You can choose the timestack, the 
real corresponding bathymetry, the cnn model/weights and autoencoder weights
for the bathymetry prediction.
"""


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ex = ApplicationWindow.ApplicationWindow(title="Bathymetry Estimation")
    sys.exit(app.exec_())
