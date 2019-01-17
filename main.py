import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from time import sleep

from map import Map

ESC = 27
X = 40
Y = 40

BODY = 1

def main(stdscr):
    # do not wait for input when calling getch
    stdscr.nodelay(1)

    map = Map(X, Y)
    map[X//2, Y//2] = BODY
    key = KEY_RIGHT

    while True:
        # get keyboard input, returns -1 if none available
        c = stdscr.getch()
        if c != -1:
            # print numeric value
            text = ' '
            if c == KEY_RIGHT:
                text = "right"
            elif c == KEY_LEFT:
                text = "left"
            elif c == KEY_DOWN:
                text = "down"
            elif c == KEY_UP:
                text = "up"
            elif c == ESC:
                break

            stdscr.addstr(str(c) + ' ' + text + '   ')
            stdscr.refresh()
            stdscr.move(0, 0)

        sleep(0.1)

if __name__ == '__main__':
    curses.wrapper(main)