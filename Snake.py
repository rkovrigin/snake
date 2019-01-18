from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN


class Cell(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Snake(object):

    def __init__(self, x, y):
        self.body = list()
        self.body.append(Cell(x, y))
        self.grow_from_fruit = 0
        self.current_direction = None

    def eat_fruit(self):
        self.grow_from_fruit += 2

    def check_crash(self, X, Y):
        head = self.body[len(self.body)-1]

        if head.x < 0 or head.x >= X:
            return False
        if head.y < 0 or head.y >= Y:
            return False

        for i in range(len(self.body)-2):
            if head.x == self.body[i].x or head.y == self.body[i].y:
                return False

        return True

    def move(self, direction=None):
        if direction not in [KEY_DOWN, KEY_UP, KEY_LEFT, KEY_RIGHT]:
            return


        self.current_direction = direction

        if self.grow_from_fruit == 0:
            for i in range(len(self.body)-1):
                self.body[i] = self.body[i+1]
        else:
            self.body.append(self.body[len(self.body)-1])
            self.grow_from_fruit -= 1

        if direction == KEY_RIGHT:
            self.body[len(self.body)-1].x += 1
        elif direction == KEY_LEFT:
            self.body[len(self.body)-1].x -= 1
        elif direction == KEY_UP:
            self.body[len(self.body)-1].y -= 1
        elif direction == KEY_DOWN:
            self.body[len(self.body)-1].y += 1

        print(self.body[len(self.body)-1].x, self.body[len(self.body)-1].y)

    def __len__(self):
        return len(self.body)

