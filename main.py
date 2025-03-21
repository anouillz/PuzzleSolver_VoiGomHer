import cv2
from matplotlib import pyplot as plt

from contours import find_piece_contours
from corners import find_corners
from piece import Piece
import numpy as np

PATH = "res/24.jpg"
#PATH = "res/49.jpg"

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
        piece = Piece(i, contour, box, corners, edges, edge_contours,img)
        piece.draw(img)
        pieces.append(piece)

    edges_d = []
    form_d = []
    for i,p in enumerate(pieces):
        for p2 in pieces[i + 1:]:
            for e in p.edges:
                for e2 in p2.edges:
                    e.matches_with(e2)
                    if e.type == e2.type:
                        continue
                    v = e2.shape_vec[::-1]
                    form_d.append((p.id, p2.id, e.idx, e2.idx, np.std(e.shape_vec + v)))
                    #print("pieces: p1: ", p.id, ", p2: ", p2.id)
                    #print("edges: p1 edge", e.idx, ", p2 edge: ", e2.idx)
                    edges_d.append(abs(e.length - e2.length) / e.length)
                    #plt.scatter(range(e.SHAPE_VEC_SIZE), e.shape_vec + v, label=f"Edge {e2.idx}")
                #plt.title(f"P{p.id}-E{e.idx} - P{p2.id}")
                #plt.legend()
                #plt.show()
                #input()

    print("min : ", np.min(edges_d))
    print("max : ", np.max(edges_d))
    print("médian : ", np.median(edges_d))
    print("moyenne : ", np.mean(edges_d))
    form_d = sorted(form_d, key=lambda x: x[4])
    for f in form_d:
        print(f)

    cv2.imshow("Pieces", img)
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()
