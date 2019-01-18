import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from time import sleep
from Snake import Snake

from map import Map

ESC = 27
X = 40
Y = 40

BODY = 1

def main(stdscr):
    stdscr.nodelay(1)
    snake = Snake(X//2, Y//2)
    key = None
    i = 0
    c = stdscr.getch()

    while c != 27:
        i += 1
        if c in [KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN]:
            key = c

        snake.move(key)

        if not snake.check_crash(X, Y):
            break

        # stdscr.addstr(str(c) + ' ' + str(len(snake)))
        stdscr.refresh()
        stdscr.move(0, 0)
        c = stdscr.getch()

        sleep(0.1)

if __name__ == '__main__':
    curses.wrapper(main)