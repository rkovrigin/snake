import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from time import sleep
from Snake import Snake

ESC = 27
X = 40
Y = 40

BODY = 1


def main(stdscr):
    stdscr.nodelay(1)
    snake = Snake(X // 2, Y // 2)
    direction = None
    i = 0
    key = stdscr.getch()

    while key != 27:
        i += 1
        stdscr.addstr(str(key))
        if key in [KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN]:
            if key == KEY_UP and direction == KEY_DOWN or \
                    key == KEY_DOWN and direction == KEY_UP or \
                    key == KEY_LEFT and direction == KEY_RIGHT or \
                    key == KEY_RIGHT and direction == KEY_LEFT:
                key = direction
                continue
            else:
                direction = key

        snake.move(direction)

        if not snake.check_crash(X, Y):
            break

        stdscr.refresh()
        stdscr.move(0, 0)
        sleep(0.1)
        key = stdscr.getch()


if __name__ == '__main__':
    curses.wrapper(main)
