import os
import random
import struct
import sys
import threading
import time
import itertools
import wave

import numpy as np
import pyaudio
import xlsxwriter
from PyQt4 import QtGui

from comparisonView import ComparisonView

__author__ = 'Andres'

"""Script to run at the lab at the acoustics research institute. The equipment require some special treatment (16kHz sampling rate is not supported, and when selecting 1 channel the sound comes from one side only.""" 


class PairWiseModel(object):
    REPETITIONS = 80

    def __init__(self, comparison_directories_list):
        self._sampling_rate = 48000
        self.p = pyaudio.PyAudio()
        self._stream = self.p.open(format=8, channels=2, rate=self._sampling_rate, output=True, output_device_index=24)
        self._directories = comparison_directories_list
        self._selections = []
        self._savedCombinations = np.array([])
        self._generateCombinations()
        self._totalNeeded = len(self._savedCombinations)
        self.initializeView()

    def initializeView(self):
        self._view = ComparisonView(self)
        self._view.show()
        self._view.updateGeometry()
        self._view.Progress.emit(len(self._selections))

        thread = threading.Thread(target=self._playNewSounds)
        thread.start()

    def _generateCombinations(self):
        self._savedCombinations = np.array(list(itertools.combinations(self._directories, 2)) * self.REPETITIONS)
        np.random.shuffle(self._savedCombinations)

    def printStats(self):
        print(len(self._selections))
        for directory in self._directories:
            wins, count = self.evaluateFor(directory)
            try:
                print('count ', directory, ':', str(len(count)))
                print('wins ', directory, ':', str(len(wins)))
                print('win % ', directory, ':', str(len(wins) / len(count)))
            except ZeroDivisionError:
                print("You need to listen to more samples")

    def saveStats(self):
        workbookName = "pairwisetest"
        for directory in self._directories:
            workbookName += "_" + directory
        workbookName += ".xlsx"

        workbook = xlsxwriter.Workbook(workbookName)
        worksheet = workbook.add_worksheet('comparisons')

        worksheet.write(0, 0, "A")
        worksheet.write(0, 1, "B")

        worksheet.write(0, 2, "Winner")

        row = 1
        for index, selection in enumerate(self._selections):
            worksheet.write(row, 0, self._savedCombinations[index, 0])
            worksheet.write(row, 1, self._savedCombinations[index, 1])
            worksheet.write(row, 2, selection)
            row += 1
        workbook.close()

    def evaluateFor(self, name):
        wins = [x for x in self._selections if x == name]
        count = [x for x in self._savedCombinations[:len(self._selections)] if name in x]
        return wins, count

    def _getNewSounds(self):
        roundsCombination = list(self._savedCombinations[len(self._selections)])
        np.random.shuffle(roundsCombination)
        self.sounds = np.array(
            [[directory, self.loadRandomSoundFrom(directory)] for directory in roundsCombination])

    def _playNewSounds(self):
        self._getNewSounds()
        self._playSounds()

    def _playSounds(self):
        self._view.LightFirstButton.emit()
        self.playWaveObject(self.sounds[0, 1])
        time.sleep(1)
        self._view.UnlightFirstButton.emit()

        time.sleep(0.5)

        self._view.LightSecondButton.emit()
        self.playWaveObject(self.sounds[1, 1])
        time.sleep(1)
        self._view.UnlightSecondButton.emit()
        time.sleep(0.25)

        self._view.EnableButtons.emit()

    def onFirstButtonClicked(self):
        self._view.DisableButtons.emit()
        self._selections.append(self.sounds[0, 0])
        time.sleep(0.25)
        self.restartView()

    def onSecondButtonClicked(self):
        self._view.DisableButtons.emit()
        self._selections.append(self.sounds[1, 0])
        time.sleep(0.25)
        self.restartView()

    def onRepeatButtonClicked(self):
        self._view.DisableButtons.emit()
        self.sounds[0, 1].rewind()
        self.sounds[1, 1].rewind()
        thread = threading.Thread(target=self._playSounds)
        thread.start()

    def playWaveObject(self, waveObject):
        thread = threading.Thread(target=self._playWaveObject, args=[waveObject])
        thread.start()

    def _playWaveObject(self, waveObject):
        data = [struct.unpack("<h", waveObject.readframes(1))[0] for i in range(waveObject.getnframes())]
        f = 0x7FFF / max((abs(i) for i in data))
        normalizedData = b''.join(struct.pack("<h", int(i * f)) + struct.pack("<h", int(i * f)) for i in data)
        self._stream.write(normalizedData)

    def restartView(self):
        if len(self._selections) < len(self._savedCombinations):
            self._view.DisableButtons.emit()
            self._view.Progress.emit(100 * len(self._selections) / self._totalNeeded)

            thread = threading.Thread(target=self._playNewSounds)
            thread.start()
        else:
            self._view.Completed.emit()

    def loadRandomSoundFrom(self, directory):
        random_filename = random.choice(os.listdir(directory))
        return wave.open(directory + '/' + random_filename, 'rb')

    def show(self):
        self._view.show()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    form = PairWiseModel(['derivs', 'tall', 'real', 'wavegan'])
    form.show()

    app.exec_()