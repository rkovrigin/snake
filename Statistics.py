from PyQt5.QtCore import Qt
from Snake import Cell

WALL = 1
BODY = 2
EMPTY = 0
FRUIT = 100


class Statistic(object):

    def __init__(self, view_range=5):
        self.data = list()
        self.view_range = view_range

    # TODO: delete the last line from snapshot before save this data to a file
    def snapshot(self, snake, fruit, x, y):
        head = snake.head
        snapshot = list()
        for i in range(head.x - self.view_range//2, (head.x + self.view_range//2) + 1):
            for j in range(head.y - self.view_range//2, (head.y + self.view_range//2) + 1):
                if i < 0 or i >= x:
                    snapshot.append(WALL)
                elif j < 0 or j >= y:
                    snapshot.append(WALL)
                elif Cell(i, j) in snake:
                    snapshot.append(BODY)
                elif Cell(i, j) == fruit:
                    snapshot.append(FRUIT)
                else:
                    snapshot.append(EMPTY)

        if snake.current_key == Qt.Key_Right:
            snapshot.extend([1, 0, 0, 0])
        elif snake.current_key == Qt.Key_Left:
            snapshot.extend([0, 1, 0, 0])
        elif snake.current_key == Qt.Key_Up:
            snapshot.extend([0, 0, 1, 0])
        elif snake.current_key == Qt.Key_Down:
            snapshot.extend([0, 0, 0, 1])

        snapshot.append(len(snake))

        print(snapshot)
        self.data.append(snapshot)
