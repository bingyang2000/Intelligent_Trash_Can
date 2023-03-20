from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
import sys
import test


class Ui_MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.path_img = ''

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(386, 127)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.retranslateUi(MainWindow)

        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(100, 30, 231, 21))
        self.label.setObjectName("label")
        self.label.setText("请输入要识别的图片")

        self.label1 = QtWidgets.QLabel(self.centralWidget)
        self.label1.setGeometry(QtCore.QRect(100, 30, 231, 51))
        self.label1.setObjectName("label")
        self.label1.setText("")

        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(190, 90, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("打开")

        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.openfile)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "1925050061-尹晨阳"))

    def openfile(self):
        self.path_img, _ = QFileDialog.getOpenFileName(self.centralWidget, '打开图片文件', './',
                                                       'Image Files(*.png *.jpg *.bmp *.jpeg);;ALL(*.*)')
        msg = test.reg(self.path_img)
        self.label1.setText(f'识别的垃圾为：{msg}')


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
