import sys
import time
from random import randrange

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QSurfaceFormat
from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QOpenGLWidget, QWidget
from Snake import Snake, Cell
from Statistics import Statistic


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
        painter.drawRect(x*self._scale, y*self._scale, self._scale, self._scale)

    def paint(self, painter, event, members):
        painter.fillRect(event.rect(), Qt.white)
        for x, y, color in members:
            painter.setBrush(color)
            self.drawRect(painter, x, y)


class Window(QWidget):
    def __init__(self, x, y, scale=15):
        super(Window, self).__init__()
        self.setWindowTitle("Snake")
        self.openGLLabel_commands = QLabel()
        self.openGL = GLWidget(self, x, y, scale)
        layout = QGridLayout()
        layout.addWidget(self.openGL)
        self.setLayout(layout)
        self.key = None
        self.timer = 200
        self.x = x
        self.y = y
        self.timer_id = self.startTimer(self.timer)


class SnakeGame(Window):
    def __init__(self, x, y, scale=10):
        self.snake = Snake(x//2, y//2)
        self.fruit = None
        super(SnakeGame, self).__init__(x, y, scale)
        self.statistic = Statistic()
        self.set_fruit()
        self.run = False

    def keyPressEvent(self, event):
        self.key = event.key()

    def set_fruit(self):
        self.fruit = None
        while not self.fruit:
            self.fruit = Cell(randrange(0, self.x), randrange(0, self.y))
            for cell in self.snake:
                if cell.x == self.fruit.x and cell.y == self.fruit.y:
                    self.fruit = None
                    break

    def timerEvent(self, event):
        if not self.run and self.key:
            self.run = True

        self.snake.move(self.key)
        if self.snake.collapse(x, y):
            self.fruit = None
            self.key = None
            self.setWindowTitle("Game over")
            self.killTimer(self.timer_id)
            self.run = False

            time.sleep(2)

            self.setWindowTitle("Snake")
            self.snake = Snake(self.x // 2, self.y // 2)
            self.set_fruit()
            self.timer = 200
            self.timer_id = self.startTimer(self.timer)

        if self.snake.check_fruit(self.fruit):
            self.set_fruit()
            self.killTimer(self.timer_id)
            # self.timer = max(50, self.timer - 20)
            self.timer_id = self.startTimer(self.timer)

        if self.run:
            self.statistic.snapshot(self.snake, self.fruit, self.x, self.y)

        cells = [(cell.x, cell.y, Qt.green) for cell in self.snake]
        cells.append((self.fruit.x, self.fruit.y, Qt.red))
        self.openGL.animate(cells)


if __name__ == '__main__':

    app = QApplication(sys.argv)

    fmt = QSurfaceFormat()
    fmt.setSamples(1)
    QSurfaceFormat.setDefaultFormat(fmt)
    x = 50
    y = 50
    window = SnakeGame(x, y)
    window.show()

    sys.exit(app.exec_())
