# from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from PyQt5.QtCore import Qt


class Cell(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Cell(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Cell(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Cell(self.x * other.x, self.y * other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return "[%d:%d]" % (self.x, self.y)


class Snake(object):

    def __init__(self, x, y, start_length=2):
        self.body = list()
        self.body.append(Cell(x, y))
        self.grow_from_fruit = start_length - 1 +10
        self.current_key = None

    def __iter__(self):
        return (cell for cell in self.body)

    def __len__(self):
        return len(self.body)

    def __contains__(self, item):
        return item in self.body

    def __str__(self):
        snake = "Snake: "
        for cell in self:
            snake = snake + "[%d:%d]" % (cell.x, cell.y)
        return snake

    @property
    def head(self):
        return self.body[0]

    def eat_fruit(self):
        self.grow_from_fruit += 1

    def collapse(self, X, Y):
        if len(self) == 1:
            return False

        if self.head.x < 0 or self.head.x >= X:
            return True
        if self.head.y < 0 or self.head.y >= Y:
            return True

        if self.head in self.body[3:]:
            return True

        return False

    def check_fruit(self, fruit):
        if self.head.x == fruit.x and self.head.y == fruit.y:
            self.eat_fruit()
            return True
        return False

    def move(self, key=None):
        if key not in [Qt.Key_Down, Qt.Key_Up, Qt.Key_Left, Qt.Key_Right] and not self.current_key:
            return

        if key == Qt.Key_Up and self.current_key != Qt.Key_Down or \
                key == Qt.Key_Down and self.current_key != Qt.Key_Up or \
                key == Qt.Key_Left and self.current_key != Qt.Key_Right or \
                key == Qt.Key_Right and self.current_key != Qt.Key_Left:
            self.current_key = key

        new_head = Cell(self.head.x, self.head.y)

        if self.grow_from_fruit == 0:
            self.body.pop()
        else:
            self.grow_from_fruit -= 1

        if self.current_key == Qt.Key_Right:
            new_head.x += 1
        elif self.current_key == Qt.Key_Left:
            new_head.x -= 1
        elif self.current_key == Qt.Key_Up:
            new_head.y -= 1
        elif self.current_key == Qt.Key_Down:
            new_head.y += 1

        self.body.insert(0, new_head)
