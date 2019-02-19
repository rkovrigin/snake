import sys
import time
from copy import deepcopy
from itertools import cycle
from random import randrange

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QSurfaceFormat
from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QOpenGLWidget, QWidget

import keys_mapping
from LogisticRegression import LogisticRegression
from NeuralNetwork import NeuralNetwork
from Snake import Snake, Cell
from Statistics import Statistic

DEFAULT_TIMER = 100


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

        self.run = False
        self.key = None
        self.prev_key = None
        self.prev_fruit = None

        self.auto = True

        if self.auto:
            try:
                self.ai = NeuralNetwork(file_name="dump_nn.txt", my_lambda=3)
                self.ai.w1, self.ai.w2 = self.ai.optimize()
                # self.ai = LogisticRegression(file_name="dump_nn.txt", my_lambda=1)
                t1, t2 = self.ai.optimize_thetas()
            except Exception as e:
                print(e)
        else:
            self.statistic = Statistic()

        self.fruit_list = list()
        for i in range(1, self.x-1, 2):
            for j in range(1, self.y-1, 2):
                self.fruit_list.append(Cell(j, i))
        for i in range(1, self.x-1, 2):
            for j in range(1, self.y-1, 2):
                self.fruit_list.append(Cell(i, j))

        self.fruit_reverse = deepcopy(self.fruit_list)
        self.fruit_reverse.reverse()
        self.fruits = cycle(self.fruit_reverse + self.fruit_list)
        self.set_fruit()

        self.snapshot = None

    def keyPressEvent(self, event):
        if not self.auto and self.fruit is not None:
            self.key = event.key()

    def set_fruit_(self):
        self.fruit = None
        while not self.fruit:
            f = next(self.fruits)
            if f not in self.snake:
                self.fruit = Cell(f.x, f.y)
                break

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

        if self.auto:
            if self.key:
                np_array_map = Statistic.create_map(self.snake, self.fruit, self.x, self.y)
                obstacles = Statistic.snapshot_prepare_data_1(np_array_map, self.snake.current_key)
                move = self.ai.predict(obstacles, self.ai.w1, self.ai.w2)
                self.key = keys_mapping.mapping_3_to_4(next_move=move, current_key=self.key, previous_key=self.prev_key)
            else:
                self.key = Qt.Key_Left

        if self.prev_fruit:
            self.snapshot = Statistic.create_snapshot(
                current_direction=self.snake.current_key,
                next_direction=self.key,
                snake=self.snake,
                fruit=self.prev_fruit,
                x=self.x,
                y=self.y)
            self.prev_fruit = None
        else:
            self.snapshot = Statistic.create_snapshot(
                current_direction=self.snake.current_key,
                next_direction=self.key,
                snake=self.snake,
                fruit=self.fruit,
                x=self.x,
                y=self.y)

        self.snake.move(self.key)
        if self.snake.collapse(x, y):
            self.killTimer(self.timer_id)

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
            self.prev_fruit = deepcopy(self.fruit)
            self.set_fruit()
            self.killTimer(self.timer_id)
            self.timer = max(150, self.timer - 10)
            self.timer_id = self.startTimer(self.timer)

        if self.run and not self.auto:
            self.statistic.save_snapshot(self.snapshot)

        self.prev_key = self.key

        cells = [(cell.x, cell.y, Qt.green) for cell in self.snake]
        cells.append((self.fruit.x, self.fruit.y, Qt.red))
        self.openGL.animate(cells)


if __name__ == '__main__':

    app = QApplication(sys.argv)

    fmt = QSurfaceFormat()
    fmt.setSamples(1)
    QSurfaceFormat.setDefaultFormat(fmt)
    x = 30
    y = 30
    window = SnakeGame(x, y)
    window.show()

    sys.exit(app.exec_())
