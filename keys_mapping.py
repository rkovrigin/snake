from PyQt5.QtCore import Qt

forward = 0
left = 1
right = 2


def mapping_3_to_4(next_move, current_key, previous_key):
    if next_move == forward:
        if current_key:
            return previous_key
        else:
            return Qt.Key_Right
    elif next_move == left:
        if current_key == Qt.Key_Up:
            return Qt.Key_Left
        elif current_key == Qt.Key_Down:
            return Qt.Key_Right
        elif current_key == Qt.Key_Right:
            return Qt.Key_Up
        elif current_key == Qt.Key_Left:
            return Qt.Key_Down
    elif next_move == right:
        if current_key == Qt.Key_Up:
            return Qt.Key_Right
        elif current_key == Qt.Key_Down:
            return Qt.Key_Left
        elif current_key == Qt.Key_Right:
            return Qt.Key_Down
        elif current_key == Qt.Key_Left:
            return Qt.Key_Up


def mapping_4_to_3(cur_direction, next_direction):
    if cur_direction == Qt.Key_Right:
        if next_direction == Qt.Key_Down:
            return right
        elif next_direction == Qt.Key_Up:
            return left
        elif next_direction == Qt.Key_Right:
            return forward
        else:
            return  None
    elif cur_direction == Qt.Key_Left:
        if next_direction == Qt.Key_Up:
            return right
        elif next_direction == Qt.Key_Down:
            return left
        elif next_direction == Qt.Key_Left:
            return forward
        else:
            return  None
    elif cur_direction == Qt.Key_Down:
        if next_direction == Qt.Key_Right:
            return left
        elif next_direction == Qt.Key_Left:
            return right
        elif next_direction == Qt.Key_Down:
            return forward
        else:
            return None
    elif cur_direction == Qt.Key_Up:
        if next_direction == Qt.Key_Right:
            return right
        elif next_direction == Qt.Key_Left:
            return left
        elif next_direction == Qt.Key_Up:
            return forward
        else:
            return None
