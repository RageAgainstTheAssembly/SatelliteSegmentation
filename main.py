import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QMainWindow
from PyQt5.uic import loadUi
import numpy as np
import os
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *
import fastai
from PIL import Image, ImageOps, ImageFilter
from PIL.ImageQt import ImageQt
from torchvision import transforms
import pathlib



ROOT_PATH = ""
DATA_PATH = os.path.join(ROOT_PATH, "DataSet/Split")
IMAGE_PATH = os.path.join(ROOT_PATH, "images")
LABEL_PATH = os.path.join(ROOT_PATH, "labels")
BATCH_SIZE = 16
path = Path(DATA_PATH)
fnames = get_image_files(path/"images")
codes = np.array(["building", "woodland", "water", "background"])
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def label_func(fn):
    return path/"labels"/f"{fn.stem}_m{'.png'}"


model = load_learner('nALLe8.pkl')


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("gui.ui", self)
        self.browse.clicked.connect(self.select_target)
        self.startButton.clicked.connect(self.predict)

    def select_target(self):
        picData = QFileDialog.getOpenFileName(self, "Open file", r"C:\Users\Dell\PycharmProjects\SegmetationProject\DataSet\Raw\PNG_images")
        #make correct path
        self.filename.setText(picData[0])
        self.Photo.setPixmap(QtGui.QPixmap(picData[0]))
        #fname[0] = path to selected image

    def display_result(self, result):
        qImage = ImageQt(result)
        self.Photo.setPixmap(QtGui.QPixmap(qImage))

    def predict(self):
        #self.startButton.hide()
        targetPath = self.filename.text()
        predictionTuple = model.predict(targetPath)
        makeImage = transforms.ToPILImage()

        backgroundTensor = predictionTuple[2][0]
        buildingTensor = predictionTuple[2][1]
        woodlandTensor = predictionTuple[2][2]
        waterTensor = predictionTuple[2][3]

        backgroundMask = makeImage(backgroundTensor)
        buildingMask = makeImage(buildingTensor)
        woodlandMask = makeImage(woodlandTensor)
        waterMask = makeImage(waterTensor)

        original = Image.open(targetPath)
        working = original
        red = Image.new('RGBA', (512, 512), (255, 0, 0, 255))
        green = Image.new('RGBA', (512, 512), (0, 255, 0, 255))
        blue = Image.new('RGBA', (512, 512), (0, 0, 255, 255))
        working = Image.composite(red, working, buildingMask)
        working = Image.composite(green, working, woodlandMask)
        working = Image.composite(blue, working, waterMask)
        result = Image.blend(working, original, alpha=0.5)
        result.save('result.jpg')
        #display_result(self, result)
        #self.startButton.show()
        print("clicked")  # we will just print clicked when the button is pressed


def show_window():
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(mainwindow)
    widget.setFixedWidth(795)
    widget.setFixedHeight(991)
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    show_window()





