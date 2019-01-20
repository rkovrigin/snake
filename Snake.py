# from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from PyQt5.QtCore import Qt

KEY_RIGHT = Qt.Key_Right
KEY_LEFT = Qt.Key_Left
KEY_UP = Qt.Key_Up
KEY_DOWN = Qt.Key_Down


class Cell(object):

    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color


class Snake(object):

    def __init__(self, x, y, start_length=2):
        self.body = list()
        self.body.append(Cell(x, y, Qt.green))
        self.grow_from_fruit = start_length - 1
        self.current_key = None

    def __iter__(self):
        return (cell for cell in self.body)

    def eat_fruit(self):
        self.grow_from_fruit += 2

    def collapse(self, X, Y):
        head = self.body[len(self.body)-1]

        if len(self) == 1:
            return False

        if head.x < 0 or head.x >= X:
            return True
        if head.y < 0 or head.y >= Y:
            return True

        for i in range(len(self.body)-2):
            if head.x == self.body[i].x and head.y == self.body[i].y:
                return True

        return False

    def check_fruit(self, fruit):
        head = self.body[len(self.body)-1]
        if head.x == fruit.x and head.y == fruit.y:
            self.eat_fruit()
            return True
        return False

    def move(self, key=None):
        if key not in [KEY_DOWN, KEY_UP, KEY_LEFT, KEY_RIGHT] and not self.current_key:
            return

        if key == KEY_UP and self.current_key != KEY_DOWN or \
                key == KEY_DOWN and self.current_key != KEY_UP or \
                key == KEY_LEFT and self.current_key != KEY_RIGHT or \
                key == KEY_RIGHT and self.current_key != KEY_LEFT:
            self.current_key = key

        head = Cell(self.body[len(self) - 1].x, self.body[len(self) - 1].y, Qt.green)

        if self.grow_from_fruit == 0:
            self.body.pop(0)
        else:
            self.grow_from_fruit -= 1

        if self.current_key == KEY_RIGHT:
            head.x += 1
        elif self.current_key == KEY_LEFT:
            head.x -= 1
        elif self.current_key == KEY_UP:
            head.y -= 1
        elif self.current_key == KEY_DOWN:
            head.y += 1

        self.body.append(head)

        print(head.x, head.y)

    def __len__(self):
        return len(self.body)

