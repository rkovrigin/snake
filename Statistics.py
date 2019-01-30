import os

from PyQt5.QtCore import Qt
from Snake import Cell
from pathlib import Path

WALL = 1
BODY = 2
EMPTY = 0
FRUIT = 5


class Statistic(object):

    def __init__(self, view_range=10):
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

        snapshot.append(len(snake))

        if snake.current_key == Qt.Key_Right:
            snapshot.extend([1, 0, 0, 0])
        elif snake.current_key == Qt.Key_Left:
            snapshot.extend([0, 1, 0, 0])
        elif snake.current_key == Qt.Key_Up:
            snapshot.extend([0, 0, 1, 0])
        elif snake.current_key == Qt.Key_Down:
            snapshot.extend([0, 0, 0, 1])

        print(snapshot[:-4], '   ', snapshot[-4:])
        self.data.append(snapshot)

    def get_overview(self, snake, fruit, x, y):
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
        print(snapshot)
        return snapshot

    def save(self, file_name="learning2.txt"):
        f = open(file_name, "a+")
        self.data.pop() # remove last line where snake meets the wall
        for snapshot in self.data:
            f.write(str(snapshot)[1:-1] + '\n')
        f.close()
        self.data.clear()
