import sys
import time
from random import randrange

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QSurfaceFormat
from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QOpenGLWidget, QWidget

from LogisticRegression import LogisticRegression
from Snake import Snake, Cell
from Statistics import Statistic

KEYS = {
    Qt.Key_Right : "RIGHT",
    Qt.Key_Left : "LEFT",
    Qt.Key_Up : "UP",
    Qt.Key_Down : "DOWN",
    None : "None"
}

DEFAULT_TIMER = 1000

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
        self.timer = DEFAULT_TIMER
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
        self.lr = LogisticRegression("learning3.txt")
        self.auto = False
        self.key = None
        self.prev_key = None
        if self.auto:
            self.lr.optimize_thetas()

    def keyPressEvent(self, event):
        if not self.auto and self.fruit is not None:
            self.key = event.key()

    def set_fruit(self):
        self.fruit = None
        # self.fruit = Cell(x//2-2, y//2-2)
        # return
        while not self.fruit:
            self.fruit = Cell(randrange(0, self.x), randrange(0, self.y))
            for cell in self.snake:
                if cell.x == self.fruit.x and cell.y == self.fruit.y:
                    self.fruit = None
                    break

    def timerEvent(self, event):
        if not self.run and self.key:
            self.run = True

        if self.auto:
            move = self.lr.predict(self.statistic.get_overview(self.snake, self.fruit, self.x, self.y))

            if move == 0:
                self.key = Qt.Key_Right
            elif move == 1:
                self.key = Qt.Key_Left
            elif move == 2:
                self.key = Qt.Key_Up
            elif move == 3:
                self.key = Qt.Key_Down

        self.snake.move(self.key)
        if self.snake.collapse(x, y):
            self.killTimer(self.timer_id)
            if not self.auto:
                self.statistic.save()

            self.run = False
            self.fruit = None
            self.key = None
            self.prev_key = None
            self.setWindowTitle("Game over")

            time.sleep(2)


            self.setWindowTitle("Snake")
            self.snake = Snake(self.x // 2, self.y // 2)
            self.set_fruit()
            self.timer = DEFAULT_TIMER
            self.timer_id = self.startTimer(self.timer)

        if self.snake.check_fruit(self.fruit):
            self.set_fruit()
            self.killTimer(self.timer_id)
            self.timer = max(80, self.timer - 10)
            self.timer_id = self.startTimer(self.timer)

        if self.run and not self.auto:
            # self.statistic.snapshot(self.prev_key, self.snake, self.fruit, self.x, self.y)
            self.statistic.save_snapshot(self.prev_key, self.snake, self.fruit, self.x, self.y)

        print("Prev:[%s] Cur:[%s]" % (KEYS[self.prev_key], KEYS[self.key]))

        self.prev_key = self.key

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
