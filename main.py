import cv2
from matplotlib import pyplot as plt

from contours import find_piece_contours
from corners import find_corners
from piece import Piece
import numpy as np

from puzzle import Puzzle

PATH = "res/24_pieces_sweater_kevin_small.jpg"
PATH = "res/49_pieces_sweater_kevin_small2.jpg"
#PATH = "res/49.jpg"

def main():
    print("##########################################")
    print("##       Fantastic  Puzzle  Solver      ##")
    print("## by A. Gomes, L. Heredero & K. Voisin ##")
    print("##########################################")
    print()

    img = cv2.imread(PATH)

    puzzle: Puzzle = Puzzle.from_image(img)
    print(puzzle)
    for piece in puzzle.pieces:
        piece.draw(img)

    cv2.imshow("Pieces", img)
    #puzzle.show_solution()
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()
