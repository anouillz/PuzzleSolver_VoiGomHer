import cv2

from contours import find_piece_contours
from corners import find_corners
from piece import Piece

PATH = "res/24_pieces_sweater_kevin_small.jpg"
PATH = "res/49_pieces_sweater_kevin_small2.jpg"

def main():
    print("##########################################")
    print("##       Fantastic  Puzzle  Solver      ##")
    print("## by A. Gomes, L. Heredero & K. Voisin ##")
    print("##########################################")
    print()

    img = cv2.imread(PATH)

    # Contours
    print("Finding contours")
    piece_contours = find_piece_contours(img)
    n_pieces = len(piece_contours)
    print(f"{n_pieces} pieces")

    # Corners + edges
    print("Identifying corners and edges")
    pieces = []
    for i, (contour, box) in enumerate(piece_contours):
        corners, edges, edge_contours = find_corners(contour)
        piece = Piece(i, contour, box, corners, edges, edge_contours)
        piece.draw(img)
        pieces.append(piece)

    cv2.imshow("Pieces", img)
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()
