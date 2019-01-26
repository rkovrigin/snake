import os
import numpy as np


class LogisticRegression(object):

    def __init__(self, file_name=""):
        self.X = None
        self.Y = None

        if os.path.exists(file_name):
            x_data = list()
            y_data = list()
            f = open(file_name, "r+")
            for line in f:
                x_data.append([int(i) for i in line.split(", ")][:-4])
                y_data.append([int(i) for i in line.split(", ")][-4:].index(1) + 1)

            self.X = np.asarray(x_data)
            self.Y = np.asarray(y_data)



lr = LogisticRegression("learning.txt")