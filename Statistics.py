import os

import numpy as np
from PyQt5.QtCore import Qt
from Snake import Cell
from pathlib import Path
import pickle

KEYS = {
    Qt.Key_Right : "RIGHT",
    Qt.Key_Left : "LEFT",
    Qt.Key_Up : "UP",
    Qt.Key_Down : "DOWN",
    None : "None"
}

EMPTY = 0
BODY = 1
HEAD = 2
WALL = 3
FRUIT = 4

FORWARD = 0
LEFT = 1
RIGHT = 2


class Snapshot(object):
    def __init__(self, my_map, cd, pd):
        self.cur_direction = cd
        self.prev_direction = pd
        self.my_map = my_map


class Statistic(object):

    def __init__(self, view_range=10, output="dump.txt"):
        self.data = list()
        self.view_range = view_range
        self.output = output
        np.set_printoptions(threshold=np.nan)

    def get_direction(self, prev_key, snake):
        if prev_key == None:
            return FORWARD
        elif prev_key != snake.current_key:
            if prev_key == Qt.Key_Right:
                if snake.current_key == Qt.Key_Up:
                    return LEFT
                elif snake.current_key == Qt.Key_Down:
                    return RIGHT
                else:
                    return FORWARD
            elif prev_key == Qt.Key_Left:
                if snake.current_key == Qt.Key_Up:
                    return RIGHT
                elif snake.current_key == Qt.Key_Down:
                    return LEFT
                else:
                    return FORWARD
            elif prev_key == Qt.Key_Up:
                if snake.current_key == Qt.Key_Right:
                    return RIGHT
                elif snake.current_key == Qt.Key_Left:
                    return LEFT
                else:
                    return FORWARD
            elif prev_key == Qt.Key_Down:
                if snake.current_key == Qt.Key_Right:
                    return LEFT
                elif snake.current_key == Qt.Key_Left:
                    return RIGHT
                else:
                    return FORWARD
        else:
            return FORWARD

    # get_3_cells

    # TODO: delete the last line from snapshot before save this data to a file
    def snapshot(self, prev_key, snake, fruit, x, y):
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
        direction = self.get_direction(prev_key, snake)
        if direction is not None:
            snapshot.append(direction)
            print(snapshot)
            self.data.append(snapshot)

        # if snake.current_key == Qt.Key_Right:
        #     snapshot.extend([1, 0, 0, 0])
        # elif snake.current_key == Qt.Key_Left:
        #     snapshot.extend([0, 1, 0, 0])
        # elif snake.current_key == Qt.Key_Up:
        #     snapshot.extend([0, 0, 1, 0])
        # elif snake.current_key == Qt.Key_Down:
        #     snapshot.extend([0, 0, 0, 1])

    def save_snapshot(self, current_direction, next_direction, snake, fruit, x, y):
        my_map = np.zeros([x, y], dtype=int)

        for cell in snake:
            my_map[cell.x, cell.y] = BODY

        my_map[snake.head.x, snake.head.y] = HEAD
        my_map[fruit.x, fruit.y] = FRUIT

        try:
            with open(self.output, "a+") as output:
                print(my_map)
                output.write("%d;%d;%s;%d;%d\n" %
                             (x, y, [i[0] for i in my_map.reshape([x*y, 1]).tolist()], current_direction, next_direction))
        except Exception as e:
            print(e)

    def read_snapshots(self, file=None, x=None, y=None):
        if not file:
            file = self.output

        snapshots = list()

        with open(file, "r") as input_file:
            for line in input_file:
                parsed = line.split(";")

                if not x:
                    x = int(parsed[0])
                if not y:
                    y = int(parsed[1])

                if int(parsed[0]) == x and int(parsed[1]) == y:
                    my_map = np.array([int(i) for i in parsed[2].strip("[]").split(', ')]).reshape((x, y))
                    current_direction = int(parsed[3])
                    next_direction = int(parsed[4])
                    snapshot = {'x': x,
                                'y': y,
                                'map': my_map,
                                'current_direction': current_direction,
                                'next_direction': next_direction}
                    snapshots.append(snapshot)

        return snapshots

    def _get_head(self, np_array):
        x, y = np_array.shape
        for i in range(x):
            for j in range(y):
                if np_array[i, j] == HEAD:
                    return Cell(i, j)

    def _get_obsacle(self, np_array, x, y):
        X, Y = np_array.shape
        if x < 0 or x >= X or y < 0 or y >= Y:
            return WALL
        return np_array[x, y]


    def _get_surroundings(self, np_array, current_direction):
        obstacle = [0, 0, 0] # forward, left, right
        head = self._get_head(np_array)
        if current_direction == Qt.Key_Right:
            obstacle[0] = self._get_obsacle(np_array, head.x + 1, head.y)
            obstacle[1] = self._get_obsacle(np_array, head.x, head.y - 1)
            obstacle[2] = self._get_obsacle(np_array, head.x, head.y + 1)
        elif current_direction == Qt.Key_Left:
            obstacle[0] = self._get_obsacle(np_array, head.x - 1, head.y)
            obstacle[1] = self._get_obsacle(np_array, head.x, head.y + 1)
            obstacle[2] = self._get_obsacle(np_array, head.x, head.y - 1)
        elif current_direction == Qt.Key_Up:
            obstacle[0] = self._get_obsacle(np_array, head.x, head.y - 1)
            obstacle[1] = self._get_obsacle(np_array, head.x - 1, head.y)
            obstacle[2] = self._get_obsacle(np_array, head.x + 1, head.y)
        elif current_direction == Qt.Key_Down:
            obstacle[0] = self._get_obsacle(np_array, head.x, head.y + 1)
            obstacle[1] = self._get_obsacle(np_array, head.x + 1, head.y)
            obstacle[2] = self._get_obsacle(np_array, head.x - 1, head.y)


        print(np_array)
        print(obstacle, KEYS[current_direction])

        return obstacle

    def prepare_data_1(self):
        data = self.read_snapshots()
        i = 0
        for snapshot in data:
            obstacles = self._get_surroundings(np_array=snapshot["map"], current_direction=snapshot["current_direction"])
            # print(self._get_head(snapshot["map"]), obstacles)
            # i += 1
            # if i >= 2:
            #     exit(1)


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

    def save(self, file_name="learning3.txt"):
        f = open(file_name, "a+")
        self.data.pop() # remove last line where snake meets the wall
        for snapshot in self.data:
            f.write(str(snapshot)[1:-1] + '\n')
        f.close()
        self.data.clear()

st = Statistic()
st.prepare_data_1()