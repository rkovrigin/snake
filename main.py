import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from time import sleep
from snake import Snake

from map import Map

ESC = 27
X = 40
Y = 40

BODY = 1

def main(stdscr):
    # do not wait for input when calling getch
    stdscr.nodelay(1)

    key = KEY_RIGHT

    snake = Snake(X//2, Y//2)

    while True:
        # get keyboard input, returns -1 if none available
        c = stdscr.getch()
        if c != -1:
            # print numeric value
            text = ' '

            snake.move(c)

            stdscr.addstr(str(c) + ' ' + str(len(snake)))
            stdscr.refresh()
            stdscr.move(0, 0)

        sleep(0.1)

if __name__ == '__main__':
    curses.wrapper(main)