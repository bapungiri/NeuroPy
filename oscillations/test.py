import matplotlib

# Make sure that we are using QT5
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication([])

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0, 0, 0, 0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        self.show()
        exit(self.qapp.exec_())


# create a figure and some subplots
fig, axes = plt.subplots(ncols=4, nrows=5, figsize=(100, 100))
for ax in axes.flatten():
    ax.plot([2, 3, 5, 1])

# pass the figure to the custom window
a = ScrollableWindow(fig)


# def main():
#     if not QtWidgets.QApplication.instance():
#         app = QtWidgets.QApplication(sys.argv)
#     else:
#         app = QtWidgets.QApplication.instance()
#     # app = QtWidgets.QApplication(sys.argv)
#     main = ScrollableWindow()
#     main.show()
#     # sys.exit(app.exec_())
#     return main


# if __name__ == "__main__":
#     m = main()
