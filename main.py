import cv2
from matplotlib import pyplot as plt

from contours import find_piece_contours
from corners import find_corners
from piece import Piece,PieceType
from edge import Edge,EdgeType
import numpy as np

from puzzle import Puzzle

PATH = "res/24_pieces_sweater_kevin_small.jpg"
#PATH = "res/49_pieces_sweater_kevin_small2.jpg"
#PATH = "res/49.jpg"

def find_pieces_matches(pieces:list[Piece],debug=True):
    result = [[() for i in range(0,len(pieces))] for i in range(0,len(pieces))]

    cornerPiece:Piece = None
    # find first corner
    for p in pieces:
        if p.type == PieceType.CORNER:
            cornerPiece = p
            break
    if not cornerPiece:
        if debug:
            print("no first corner found")
        else:
            print("first corner found",cornerPiece)

    borderPieces = [p for p in pieces if p.type == PieceType.EDGE]

    def createBorderMatches(currentPiece:Piece,pieces:list[Piece],edgePieces:list[Piece],currentEdgeId,resultArray,x=0,y=0,directionx=0,directiony=0):
        bestMatchValue = float("-inf")
        bestMatch:Edge = None
        bestMatchId = None
        currentEdge = currentPiece.edges[currentEdgeId]
        for p in edgePieces:
            for i,e in enumerate(p.edges):
                matchValue = currentEdge.matches_with(e)
                if matchValue > bestMatchValue:
                    bestMatchValue = matchValue
                    bestMatch = e
                    bestMatchId = i
        top_id = None
        if directiony == 0:
            top_id = bestMatchId - directionx
        elif directionx == 0:
            top_id = bestMatchId
        top_id = top_id % 4


        new_match = (pieces.index(bestMatch),top_id)
        resultArray[y][x] = new_match

        if x >= len(resultArray):



    firstEdgeIndex = None
    for i,e in enumerate(cornerPiece.edges):
        if e.type == EdgeType.INSIDE or e.type == EdgeType.OUTSIDE:
            firstEdgeIndex = i
            break
    if not firstEdgeIndex:
        if debug:
            print("no first edge index")

    createBorderMatches(cornerPiece,borderPieces,firstEdgeIndex,result)














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

    find_pieces_matches(puzzle.pieces)


if __name__ == '__main__':
    main()
