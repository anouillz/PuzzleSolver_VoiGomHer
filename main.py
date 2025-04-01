import cv2
from matplotlib import pyplot as plt

from contours import find_piece_contours
from corners import find_corners
from piece import Piece
import numpy as np

from puzzle import Puzzle

PATH = "res/24_pieces_sweater_kevin_small.jpg"
#PATH = "res/49_pieces_sweater_kevin_small2.jpg"
#PATH = "res/49.jpg"

def main():
    print("##########################################")
    print("##       Fantastic  Puzzle  Solver      ##")
    print("## by A. Gomes, L. Heredero & K. Voisin ##")
    print("##########################################")
    print()

    img = cv2.imread(PATH)

    puzzle: Puzzle = Puzzle.from_image(img)
    puzzle.solution = np.array([
        [
            [22, 2],
            [0, 1],
            [6, 0],
            [20, 2],
            [7, 3],
            [11, 3],
        ],
        [
            [21, 0],
            [17, 2],
            [5, 1],
            [4, 0],
            [1, 2],
            [3, 2],
        ],
        [
            [12, 2],
            [10, 0],
            [19, 1],
            [14, 0],
            [2, 1],
            [8, 0],
        ],
        [
            [16, 1],
            [13, 1],
            [15, 3],
            [9, 2],
            [23, 3],
            [18, 1]
        ]
    ])
    print(puzzle)
    for piece in puzzle.pieces:
        #cv2.imshow("a", piece.get_rotated_img(0))
        piece.draw(img)
        #cv2.waitKey(-1)

    cv2.imshow("Pieces", img)
    puzzle.show_solution()
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()
