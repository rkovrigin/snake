import curses
import sys
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from time import sleep

from PyQt5.QtWidgets import QApplication

from Snake import Snake
from opengl_animation import WorldWindow

ESC = 27
X = 20
Y = 60

BODY = 1


def main(stdscr):
    stdscr.nodelay(1)
    snake = Snake(X // 2, Y // 2)
    key = None

    while key != 27:
        stdscr.addstr(str(key))

        snake.move(key)
        if not snake.check_crash(X, Y):
            break

        stdscr.refresh()
        stdscr.move(0, 0)

        sleep(0.05)
        key = stdscr.getch()


if __name__ == '__main__':
    curses.wrapper(main)
