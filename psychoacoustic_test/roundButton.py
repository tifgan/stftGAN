from PyQt4 import QtGui
from PyQt4.QtCore import QRect

from PyQt4.QtGui import QPushButton

__author__ = 'Andres'


class RoundButton(QPushButton):
    def resizeEvent(self, event):
        rectRegion = QRect(-2, -2, self.width()+3, self.height()+3)
        self.setMask(QtGui.QRegion(rectRegion, QtGui.QRegion.Ellipse))
        super().resizeEvent(event)
