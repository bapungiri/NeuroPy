from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5 import QtCore

app = QApplication([])
QtCore.QCoreApplication.instance().quit()

label = QLabel("Hello")
label.show()
app.exec_()
