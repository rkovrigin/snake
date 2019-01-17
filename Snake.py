from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN


class Cell(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Snake(object):

    def __init__(self, x, y):
        self.body = list(Cell(x, y))
        self.grow_from_fruit = 0

    def eat_fruit(self):
        self.grow_from_fruit += 2

    def move(self, direction=None):
        if not direction or direction not in [KEY_DOWN, KEY_UP, KEY_LEFT, KEY_RIGHT]:
            return

        if self.grow_from_fruit == 0:
            for i in range(len(self.body)-1):
                self.body[i] = self.body[i+1]
        else:
            self.body.append(self.body[len(self.body)-1])

        if self.direction == KEY_RIGHT:
            self.body[len(self.body)-1].x += 1
        elif self.direction == KEY_LEFT:
            self.body[len(self.body)-1].x -= 1
        elif self.direction == KEY_UP:
            self.body[len(self.body)-1].y -= 1
        elif self.direction == KEY_DOWN:
            self.body[len(self.body)-1].x += 1

    def __len__(self):
        return len(self.body)

