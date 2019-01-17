import numpy as np

class Wall(object):
    pass

class Map(object):

    def __init__(self, X, Y, wrapper_x=False, wrapper_y=False):
        self._X = X
        self._Y = Y
        self._wrapper_x = wrapper_x
        self._wrapper_y = wrapper_y
        self._map = np.empty([X, Y], dtype=int)

    def at(self, x, y):
        if self._wrapper_x:
            x = x % self._X
        if y >= self._Y:
            y = y % self._Y

        if x < 0 or x >= self._X:
            return Wall()

        if y < 0 or y >= self._Y:
            return Wall()

        return self._map[x, y]

    def right(self, x, y):
        self._map[x, y] = 0
        self._map[x+1, y] = 0