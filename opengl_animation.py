import sys
import time
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QSurfaceFormat
from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QOpenGLWidget, QWidget
from Snake import Snake


class GLWidget(QOpenGLWidget):
    def __init__(self, parent, x, y, scale):
        super(GLWidget, self).__init__(parent)

        self.elapsed = 0
        self._x = x
        self._y = y
        self._scale = scale
        self._parent = parent
        self.setFixedSize(x*scale, y*scale)
        self.setAutoFillBackground(False)
        self.background = QBrush(QColor(255, 255, 255))
        self._members = None
        self._click_x = None
        self._click_y = None
        self.painter = QPainter()
        self.painterImg = QPainter()
        self.count = 0

    def animate(self, members):
        self._members = members
        self.elapsed = 1
        self.update()

    def paintEvent(self, event):
        self.painter.begin(self)
        if self._members:
            self.paint(self.painter, event, self._members)
        self.painter.end()

    def drawRect(self, painter, x, y):
        painter.setBrush(Qt.green)
        painter.drawRect(x*self._scale, y*self._scale, self._scale, self._scale)

    def paint(self, painter, event, members):
        painter.fillRect(event.rect(), Qt.white)
        for cell in members:
            self.drawRect(painter, cell.x, cell.y)


class Window(QWidget):
    def __init__(self, x, y, scale=5):
        super(Window, self).__init__()
        self.setWindowTitle("Snake")
        self.openGLLabel_commands = QLabel()
        self.openGL = GLWidget(self, x, y, scale)
        layout = QGridLayout()
        layout.addWidget(self.openGL)
        self.setLayout(layout)
        self.key = None
        self.timer = 200
        self.startTimer(self.timer)
        self.x = x
        self.y = y
        self.snake = Snake(x//2, y//2)

    def keyPressEvent(self, event):
        self.key = event.key()

    def timerEvent(self, event):
        self.snake.move(self.key)
        if self.snake.collapse(x, y):
            self.setWindowTitle("Game over")
            self.snake = Snake(self.x//2, self.y//2)
            self.key = None
            time.sleep(1)
            self.setWindowTitle("Snake")
        self.openGL.animate([cell for cell in self.snake])


if __name__ == '__main__':

    app = QApplication(sys.argv)

    fmt = QSurfaceFormat()
    fmt.setSamples(1)
    QSurfaceFormat.setDefaultFormat(fmt)
    x = 60
    y = 40
    window = Window(x, y)
    window.show()

    sys.exit(app.exec_())
