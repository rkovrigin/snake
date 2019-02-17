import math
from math import sqrt, acos

import numpy as np
from PyQt5.QtCore import Qt

import keys_mapping
from Snake import Cell

KEYS = {
    Qt.Key_Right: "RIGHT",
    Qt.Key_Left: "LEFT",
    Qt.Key_Up: "UP",
    Qt.Key_Down: "DOWN",
    None: "None"
}

EMPTY = 0
BODY = 1
HEAD = 2
WALL = 3
FRUIT = 4

ITEM = 1

FORWARD = 0
LEFT = 1
RIGHT = 2


class Snapshot(object):
    def __init__(self, my_map, cd, pd):
        self.cur_direction = cd
        self.prev_direction = pd
        self.my_map = my_map


class Statistic(object):

    def __init__(self, view_range=10, output="dump.txt", write=False):
        self.data = list()
        self.view_range = view_range
        self.output = output
        np.set_printoptions(threshold=np.nan)
        self.file = None

    def get_direction(self, prev_key, current_key, snake):
        if prev_key is None:
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

    def snapshot(self, prev_key, snake, fruit, x, y):
        head = snake.head
        snapshot = list()
        for i in range(head.x - self.view_range // 2, (head.x + self.view_range // 2) + 1):
            for j in range(head.y - self.view_range // 2, (head.y + self.view_range // 2) + 1):
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
            # print(snapshot)
            self.data.append(snapshot)

    def create_map(self, snake, fruit, x, y):
        my_map = np.zeros([x, y], dtype=int)
        for cell in snake:
            my_map[cell.x, cell.y] = BODY
        my_map[snake.head.x, snake.head.y] = HEAD
        my_map[fruit.x, fruit.y] = FRUIT

        if snake.head == fruit:
            my_map[snake.head.x, snake.head.y] = HEAD + FRUIT

        return my_map

    def print_map(self, snake, fruit, x, y):
        my_map = self.create_map(snake, fruit, x, y)
        print(np.flip(np.rot90(my_map), 0))

    def create_snapshot(self, current_direction, next_direction, snake, fruit, x, y):
        if not fruit or not current_direction or not next_direction:
            return

        my_map = self.create_map(snake, fruit, x, y)

        try:
            return ("%d;%d;%s;%d;%d\n" %
                             (x, y, [i[0] for i in my_map.reshape([x * y, 1]).tolist()], current_direction,
                              next_direction))
        except Exception as e:
            print(e)

    def save_snapshot(self, snapshot):
        if not snapshot:
            return

        if not self.file:
            self.file = open(self.output, "a+")

        self.file.write(snapshot)
        self.file.flush()

    def read_snapshots(self, file=None, x=None, y=None):
        if not file:
            file = self.output

        snapshots = list()

        with open(file, "r") as input_file:
            for line in input_file:
                parsed = line.split(";")
                x = int(parsed[0])
                y = int(parsed[1])
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

    def _get_first(self, np_array, value):
        x, y = np_array.shape
        for i in range(x):
            for j in range(y):
                if np_array[i, j] == value:
                    return Cell(i, j)

    def _get_head(self, np_array):
        head = self._get_first(np_array, HEAD)
        if not head:
            return self._get_first(np_array, HEAD + FRUIT)
        return head

    def _get_fruit(self, np_array):
        fruit = self._get_first(np_array, FRUIT)
        if not fruit:
            return self._get_first(np_array, FRUIT + HEAD)
        return fruit

    def _get_obsacle(self, np_array, x, y):
        X, Y = np_array.shape
        if x < 0 or x >= X or y < 0 or y >= Y:
            return WALL
        return np_array[x, y]

    def _is_element_on_my_way(self, np_array, x, y, item=WALL):
        _x, _y = np_array.shape
        if x < 0 or x >= _x or y < 0 or y >= _y:
            if item == WALL:
                return ITEM
            else:
                return EMPTY
        if item == WALL:
            if np_array[x, y] == BODY or np_array[x, y] == WALL:
                return ITEM
        elif item == FRUIT:
            if np_array[x, y] == FRUIT:
                return 1
        return EMPTY

    def _get_surroundings(self, np_array, current_direction, item=WALL):
        # TODO: Qt.Key_left - change left and right - DONE
        obstacle = [0, 0, 0]  # forward, left, right
        head = self._get_head(np_array)
        if current_direction == Qt.Key_Right:
            obstacle[0] = self._is_element_on_my_way(np_array, head.x + 1, head.y, item=item)
            obstacle[1] = self._is_element_on_my_way(np_array, head.x, head.y - 1, item=item)
            obstacle[2] = self._is_element_on_my_way(np_array, head.x, head.y + 1, item=item)
        elif current_direction == Qt.Key_Left:
            obstacle[0] = self._is_element_on_my_way(np_array, head.x - 1, head.y, item=item)
            obstacle[1] = self._is_element_on_my_way(np_array, head.x, head.y + 1, item=item)
            obstacle[2] = self._is_element_on_my_way(np_array, head.x, head.y - 1, item=item)
        elif current_direction == Qt.Key_Up:
            obstacle[0] = self._is_element_on_my_way(np_array, head.x, head.y - 1, item=item)
            obstacle[1] = self._is_element_on_my_way(np_array, head.x - 1, head.y, item=item)
            obstacle[2] = self._is_element_on_my_way(np_array, head.x + 1, head.y, item=item)
        elif current_direction == Qt.Key_Down:
            obstacle[0] = self._is_element_on_my_way(np_array, head.x, head.y + 1, item=item)
            obstacle[1] = self._is_element_on_my_way(np_array, head.x + 1, head.y, item=item)
            obstacle[2] = self._is_element_on_my_way(np_array, head.x - 1, head.y, item=item)
        return obstacle

    def _calc_distance(self, head, fruit):
        return sqrt((head.x - fruit.x)**2 + (head.y - fruit.y)**2)

    def _calc_angle_rad(self, head, fruit):
        return math.atan2(head.y - fruit.y, head.x - fruit.x)

    def _calc_angle_degrees(self, head, fruit):
        angle = math.degrees(self._calc_angle_rad(head, fruit))
        # if angle < 0:
        #     angle += 360
        return angle


    def _print_user_friendly(self, np_array):
        print(np.rot90(np.flip(np_array, 1)))

    def _get_snake_length(self, np_array):
        x, y = np_array.shape
        length = 0
        for i in range(x):
            for j in range(y):
                if np_array[i, j] == HEAD or np_array[i, j] == BODY or np_array[i, j] == HEAD + FRUIT:
                    length += 1
        return length

    def snapshot_prepare_data_1(self, np_array, current_key):
        x, y = np_array.shape
        head = self._get_head(np_array)
        fruit = self._get_fruit(np_array)

        wall_on_my_way = self._get_surroundings(np_array=np_array, current_direction=current_key, item=WALL)
        fruit_on_my_way = self._get_surroundings(np_array=np_array, current_direction=current_key, item=FRUIT)
        obstacles = wall_on_my_way + fruit_on_my_way

        diag = (math.sqrt(x**2 + y**2))
        obstacles.append((self._calc_distance(head, fruit)) / diag)
        obstacles.append((self._calc_distance(fruit, head)) / diag)
        obstacles.append(self._calc_angle_degrees(head, fruit) / 180)
        obstacles.append(self._calc_angle_degrees(fruit, head) / 180)
        # obstacles.append(math.fabs(head.x - fruit.x) / x)
        # obstacles.append(math.fabs(head.y - fruit.y) / y)

        obstacles.append((head.x - fruit.x) / diag)
        obstacles.append((head.y - fruit.y) / diag)
        obstacles.append((fruit.x - head.x) / diag)
        obstacles.append((fruit.y - head.y) / diag)

        key = current_key - 16777234
        obstacles.append(key / 4)
        return obstacles

    def prepare_data_1(self):
        data = self.read_snapshots()
        learning_data = list()
        for snapshot in data:
            obstacles = self.snapshot_prepare_data_1(np_array=snapshot["map"], current_key=snapshot["current_direction"])
            next_direction = keys_mapping.mapping_4_to_3(snapshot["current_direction"], snapshot["next_direction"])
            obstacles.append(next_direction)
            if next_direction is not None:
                print(obstacles)
                learning_data.append(obstacles)
        return learning_data

    def get_overview(self, snake, fruit, x, y):
        head = snake.head
        snapshot = list()
        for i in range(head.x - self.view_range // 2, (head.x + self.view_range // 2) + 1):
            for j in range(head.y - self.view_range // 2, (head.y + self.view_range // 2) + 1):
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
        # print(snapshot)
        return snapshot

    def save(self, file_name="learning3.txt"):
        f = open(file_name, "a+")
        self.data.pop()  # remove last line where snake meets the wall
        for snapshot in self.data:
            f.write(str(snapshot)[1:-1] + '\n')
        f.close()
        self.data.clear()


def main():
    st = Statistic()
    # data = st.prepare_data_1()
    # np.set_printoptions(threshold=np.nan)
    # for snapshot in data:
    #     st._print_user_friendly(snapshot["map"])

    with open("dump.txt", "r") as input_file:
        for line in input_file:
            parsed = line.split(";")
            x = int(parsed[0])
            y = int(parsed[1])
            my_map = np.array([int(i) for i in parsed[2].strip("[]").split(', ')]).reshape((x, y))
            st._print_user_friendly(my_map)
            print('\n\n')


if __name__ == "__main__":
    main()
