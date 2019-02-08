from PyQt4 import QtCore, QtGui, Qt
from PyQt4.QtGui import QPushButton, QIcon
from PyQt4.QtCore import pyqtSignal, QSize

from roundButton import RoundButton
from stylesheets import ButtonsStylesheet, UnabledButtonsStylesheet, LightedButtonsStylesheet

__author__ = 'Andres'

class ComparisonView(QtGui.QMainWindow):
    LightFirstButton = pyqtSignal()
    LightSecondButton = pyqtSignal()
    UnlightFirstButton = pyqtSignal()
    UnlightSecondButton = pyqtSignal()
    EnableButtons = pyqtSignal()
    DisableButtons = pyqtSignal()
    Progress = pyqtSignal(int)
    Completed = pyqtSignal()

    def __init__(self, model, parent=None):
        super(ComparisonView, self).__init__(parent)
        self._model = model
        self.setupUi()

    def setupUi(self):
        self.showFullScreen()
        font = QtGui.QFont("Times New Roman")
        font.setPixelSize(24)
        font.setBold(True)
        self.setFont(font)
        self.setAutoFillBackground(False)
        self.centralwidget = QtGui.QWidget(self)
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setSpacing(24)

        self.counterLabel = QtGui.QLabel()

        self.questionLabel = QtGui.QLabel("Which sample do you prefer?")

        self.firstButton = RoundButton()
        icon = QIcon("speaker-512.png")
        self.firstButton.setIcon(icon)
        self.firstButton.setIconSize(QSize(100, 100))
        self.firstButton.setStyleSheet(UnabledButtonsStylesheet)
        self.secondButton = RoundButton()
        self.secondButton.setIcon(icon)
        self.secondButton.setIconSize(QSize(100, 100))
        self.secondButton.setStyleSheet(UnabledButtonsStylesheet)
        self.firstButton.clicked.connect(self._model.onFirstButtonClicked)
        self.secondButton.clicked.connect(self._model.onSecondButtonClicked)
        self.firstButton.setEnabled(False)
        self.secondButton.setEnabled(False)

        self.repeatButton = RoundButton()
        repeatIcon = QIcon("repeat.png")
        self.repeatButton.setIcon(repeatIcon)
        self.repeatButton.setIconSize(QSize(100, 100))
        self.repeatButton.setStyleSheet(UnabledButtonsStylesheet)
        self.repeatButton.clicked.connect(self._model.onRepeatButtonClicked)
        self.repeatButton.setEnabled(False)

        self.EnableButtons.connect(self._enableButtons)
        self.DisableButtons.connect(self._disableButtons)

        # self.printButton = QPushButton()
        # icon = QIcon("print.png")
        # self.printButton.setIcon(icon)
        # self.printButton.setIconSize(QSize(100, 100))
        # self.printButton.setStyleSheet(ButtonsStylesheet)
        # self.printButton.clicked.connect(self.onPrintButtonClicked)

        self.gridLayout.addWidget(self.counterLabel, 0, 0, 1, 3, QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.questionLabel, 1, 0, 1, 3, QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.firstButton, 2, 0, 1, 1, QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.repeatButton, 2, 1, 1, 1, QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.secondButton, 2, 2, 1, 1, QtCore.Qt.AlignCenter)
        # self.gridLayout.addWidget(self.printButton, 3, 0, 1, 3, QtCore.Qt.AlignCenter)

        self.setCentralWidget(self.centralwidget)

        self.LightFirstButton.connect(self.ligthUpFirstButton)
        self.LightSecondButton.connect(self.ligthUpSecondButton)
        self.UnlightFirstButton.connect(self.unlightFirstButton)
        self.UnlightSecondButton.connect(self.unlightSecondButton)
        self.Progress.connect(self.setProgress)
        self.Completed.connect(self._completed)

        self.updateGeometry()

    # def onPrintButtonClicked(self):
    #     self._model.printStats()

    def _enableButtons(self):
        self.firstButton.setEnabled(True)
        self.secondButton.setEnabled(True)
        self.repeatButton.setEnabled(True)
        self.firstButton.setStyleSheet(ButtonsStylesheet)
        self.secondButton.setStyleSheet(ButtonsStylesheet)
        self.repeatButton.setStyleSheet(ButtonsStylesheet)

    def _disableButtons(self):
        self.firstButton.setEnabled(False)
        self.secondButton.setEnabled(False)
        self.repeatButton.setEnabled(False)
        self.firstButton.setStyleSheet(UnabledButtonsStylesheet)
        self.secondButton.setStyleSheet(UnabledButtonsStylesheet)
        self.repeatButton.setStyleSheet(UnabledButtonsStylesheet)

    def ligthUpFirstButton(self):
        self.firstButton.setStyleSheet(LightedButtonsStylesheet)

    def ligthUpSecondButton(self):
        self.secondButton.setStyleSheet(LightedButtonsStylesheet)

    def unlightFirstButton(self):
        self.firstButton.setStyleSheet(UnabledButtonsStylesheet)

    def unlightSecondButton(self):
        self.secondButton.setStyleSheet(UnabledButtonsStylesheet)

    def setProgress(self, progress):
        self.counterLabel.setText("You are " + str(progress) + "% done!")

    def _completed(self):
        self.setProgress(100)
        self.firstButton.hide()
        self.secondButton.hide()
        self.repeatButton.hide()
        self.questionLabel.setText("Finished!")

    def closeEvent(self, event):
        super().closeEvent(event)
        self._model.printStats()
        self._model.saveStats()
